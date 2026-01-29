import datetime
from typing import List, Tuple
from femr import Patient, Event
from femr.extension import datasets as extension_datasets
from ehrshot.labelers.core import Label, Labeler, LabelType
from ehrshot.labelers.omop import (
    WithinVisitLabeler,
    get_inpatient_admission_events,
    move_datetime_to_end_of_day,
    get_inpatient_admission_discharge_times,
    get_femr_codes,
    identity,
)


# --- add to ehrshot/labelers/mimic.py ---------------------------------------
import datetime
from typing import List, Optional, Tuple
from femr import Patient, Event
from femr.extension import datasets as extension_datasets
from ehrshot.labelers.core import Label, Labeler, LabelType

class Mimic_ICUEventStreamMortalityLabeler(Labeler):
    """
    Event-stream version of the ICU mortality labeler:
      - Use "MIMIC/ICU_ADMISSION..." as ICU admission, paired with the first subsequent
        "MIMIC/ICU_DISCHARGE..." as ICU discharge
      - Keep only stays with ICU length >= 24h and no death within the first 24h
      - Prediction time = admission + 24h
      - The first outcome event in [prediction time, discharge] determines the label:
          * SNOMED/419620001 => label=1
          * MIMIC/ICU_DISCHARGE... => label=0
      - Skip the stay if no discharge event is found
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        icu_admit_prefix: str = "MIMIC/ICU_ADMISSION",
        icu_discharge_prefix: str = "MIMIC/ICU_DISCHARGE",
        death_codes: Tuple[str, ...] = ("SNOMED/419620001",),
        min_hours: int = 24,
    ):
        self.ontology: extension_datasets.Ontology = ontology  # Keep interface consistency; this class does not rely on ontology mapping
        self.icu_admit_prefix = icu_admit_prefix
        self.icu_discharge_prefix = icu_discharge_prefix
        self.death_codes = set(death_codes)
        self.min_hours = int(min_hours)

    def get_labeler_type(self) -> LabelType:
        return "boolean"

    # ---- Core: label based on a single patient's event stream ----
    def label(self, patient: Patient) -> List[Label]:
        evs: List[Event] = sorted(patient.events, key=lambda e: e.start)
        n = len(evs)
        i = 0
        labels: List[Label] = []

        while i < n:
            e = evs[i]
            c = getattr(e, "code", None)
            if isinstance(c, str) and c.startswith(self.icu_admit_prefix):
                start_t = e.start

                # Find the first ICU_DISCHARGE after admission as the end of this stay
                j = i + 1
                end_t: Optional[datetime.datetime] = None
                while j < n:
                    cj = getattr(evs[j], "code", None)
                    if isinstance(cj, str) and cj.startswith(self.icu_discharge_prefix):
                        end_t = evs[j].start  # Use discharge event time as stay end
                        break
                    j += 1

                # No discharge => skip this stay
                if end_t is None:
                    i += 1
                    continue

                # Duration < 24h => skip
                if (end_t - start_t).total_seconds() < self.min_hours * 3600:
                    i = j + 1
                    continue

                t_pred = start_t + datetime.timedelta(hours=self.min_hours)

                # Death within the first 24h => skip this sample
                early_death = False
                k = i + 1
                while k < j:
                    ck = getattr(evs[k], "code", None)
                    tk = evs[k].start
                    if ck in self.death_codes and tk <= t_pred:
                        early_death = True
                        break
                    k += 1
                if early_death:
                    i = j + 1
                    continue

                # Find the first outcome event (death or discharge) in [t_pred, end_t]
                first_event_code = None
                first_event_time = None
                k = i + 1
                while k <= j:  # include j (discharge)
                    ck = getattr(evs[k], "code", None)
                    tk = evs[k].start
                    if tk >= t_pred and (ck in self.death_codes or (isinstance(ck, str) and ck.startswith(self.icu_discharge_prefix))):
                        first_event_code = ck
                        first_event_time = tk
                        break
                    k += 1

                # In theory discharge should always be found; if not (edge case), conservatively skip
                if first_event_code is None:
                    i = j + 1
                    continue

                label_val = bool(first_event_code in self.death_codes)
                # Prediction time is used as the label time (downstream 2_generate_labels.py will align to minute precision)
                labels.append(Label(time=t_pred, value=label_val))

                # Continue scanning for the next admission (after this discharge)
                i = j + 1
            else:
                i += 1

        return labels
# ---------------------------------------------------------------------------

# --- add to ehrshot/labelers/mimic.py ---------------------------------------
import datetime
import json
from typing import Dict, Iterable, List, Optional, Set, Tuple

from femr import Patient, Event
from femr.extension import datasets as extension_datasets
from ehrshot.labelers.core import Label, Labeler, LabelType


class Mimic_ICUEventStreamPhenotypeLabeler(Labeler):
    """
    MIMIC-IV ICU phenotype multi-label labeler (based on OMOP/ICD9CM code mapping):

    - Use the "MIMIC/ICU_ADMISSION" event as the ICU admission trigger
    - Prediction time t_pred = admission time + min_hours (default 24h)
    - Target window = [t_pred, time of the first "MIMIC/HOSPITAL_DISCHARGE" event]
    - Within the target window, if an event code matches any phenotype's code set,
      that phenotype is marked as positive
    - Output: at most one Label per ICU stay, where value is List[str] (multi-label)
    - Skip the stay if no hospital discharge is found, or discharge time <= t_pred,
      or no phenotype is hit within the window
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        phenotype_to_codes: Dict[str, Iterable[str]],
        icu_admit_prefix: str = "MIMIC/ICU_ADMISSION",
        hosp_discharge_prefix: str = "MIMIC/HOSPITAL_DISCHARGE",
        min_hours: int = 24,
    ):
        self.ontology: extension_datasets.Ontology = ontology  # For interface consistency; this class does not use ontology mapping
        self.icu_admit_prefix = icu_admit_prefix
        self.hosp_discharge_prefix = hosp_discharge_prefix
        self.min_hours = int(min_hours)

        # Normalize mapping: phenotype -> set(codes)
        self.phenotype_to_codes: Dict[str, Set[str]] = {
            str(phe): {str(c).strip() for c in codes if str(c).strip()}
            for phe, codes in phenotype_to_codes.items()
        }
        # Reverse index: code -> set(phenotypes)
        self.code_to_phenotypes: Dict[str, Set[str]] = {}
        for phe, codes in self.phenotype_to_codes.items():
            for c in codes:
                self.code_to_phenotypes.setdefault(c, set()).add(phe)

    @classmethod
    def from_json(
        cls,
        ontology: extension_datasets.Ontology,
        json_path: str,
        **kwargs,
    ) -> "Mimic_ICUEventStreamPhenotypeLabeler":
        with open(json_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return cls(ontology=ontology, phenotype_to_codes=mapping, **kwargs)

    def get_labeler_type(self) -> LabelType:
        # If your framework uses a different name for multi-label type, change it here
        return "multilabel"

    def label(self, patient: Patient) -> List[Label]:
        evs: List[Event] = sorted(patient.events, key=lambda e: e.start)
        n = len(evs)
        i = 0
        labels: List[Label] = []

        while i < n:
            e = evs[i]
            code_i = getattr(e, "code", None)

            # Detect ICU admission trigger
            if isinstance(code_i, str) and code_i.startswith(self.icu_admit_prefix):
                start_t = e.start

                # Find the first hospital discharge after this admission as the end (entire hospitalization)
                j = i + 1
                end_t: Optional[datetime.datetime] = None
                while j < n:
                    cj = getattr(evs[j], "code", None)
                    if isinstance(cj, str) and cj.startswith(self.hosp_discharge_prefix):
                        end_t = evs[j].start
                        break
                    j += 1

                # No hospital discharge => skip this stay
                if end_t is None:
                    i += 1
                    continue

                t_pred = start_t + datetime.timedelta(hours=self.min_hours)
                # If discharge occurs before prediction time (<24h stay), target window is empty; skip
                if end_t <= t_pred:
                    i = j + 1
                    continue

                # Collect phenotype hits within [t_pred, end_t]
                phe_hits: Set[str] = set()
                k = i + 1
                while k < n:
                    ek = evs[k]
                    ck = getattr(ek, "code", None)
                    tk = ek.start
                    if tk > end_t:
                        break
                    if tk >= t_pred and isinstance(ck, str):
                        # Use full code exact match (e.g., "ICD9CM/410.71")
                        if ck in self.code_to_phenotypes:
                            phe_hits.update(self.code_to_phenotypes[ck])
                    k += 1

                # Produce a label only if at least one phenotype is hit
                if phe_hits:
                    labels.append(Label(time=t_pred, value=sorted(phe_hits)))

                # Move to the next scan segment: jump to after discharge
                i = j + 1
            else:
                i += 1

        return labels
# ---------------------------------------------------------------------------



class Mimic_ReadmissionLabeler(Labeler):
    """30-day readmissions prediction task.
    Binary prediction task @ 11:59PM on the day of discharge whether the patient will be readmitted within 30 days.
    
    Excludes:
        - Readmissions that occurred on the same day
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.time_horizon: datetime.timedelta = datetime.timedelta(days=30)
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def label(self, patient: Patient) -> List[Label]:
        labels: List[Label] = []
        times: List[Tuple[datetime.datetime]] = get_inpatient_admission_discharge_times(patient, self.ontology)
        admission_times = sorted([ x[0] for x in times ])
        for idx, admission_time in enumerate(admission_times):
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(admission_time)
            is_30_day_readmission = False
            for admission_time2 in admission_times[idx + 1:]:
                # Ignore readmissions that occur on before or on same day as prediction time
                if admission_time2 <= prediction_time:
                    continue
                # If readmission occurs within 30 days, mark as True
                if (admission_time2 - prediction_time) <= self.time_horizon:
                    is_30_day_readmission: bool = True
                    break
            labels.append(Label(prediction_time, is_30_day_readmission))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class Mimic_MortalityLabeler(WithinVisitLabeler):
    """In-hospital mortality prediction task.

    Binary prediction task @ 11:59PM on the day of admission whether the patient dies during their hospital stay.

    Excludes:
        - Admissions with no length-of-stay (i.e. `event.end is None` )
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day
        self.visit_start_adjust_func = identity
        self.visit_end_adjust_func = identity

    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Return a list of all visits.

        Excludes:
            - Admissions with no length-of-stay (i.e. `event.end is None` )
        """
        visits, visit_idxs = get_inpatient_admission_events(patient, self.ontology, is_return_idx=True)  # type: ignore
        valid_events: List[Event] = []
        for e in visits:
            if (
                e.end is not None
            ):
                valid_events.append(e)
        return valid_events

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome (i.e., death)."""
        death_concepts: List[str] = [ "SNOMED/419620001", ]
        outcome_codes: List[str] = list(get_femr_codes(self.ontology, death_concepts, is_ontology_expansion=True))
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in outcome_codes:
                times.append(e.start)
        return times

    def get_labeler_type(self) -> LabelType:
        return "boolean"

if __name__ == '__main__':
    pass