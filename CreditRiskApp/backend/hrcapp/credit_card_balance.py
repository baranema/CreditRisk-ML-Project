from typing import Optional 
from pydantic import BaseModel, Field, validator
import app_validators

POSSIBLE_NAME_CONTRACT_STATUS = ['Active', 'Completed', 'Demand', 'Signed', 'Sent proposal', 'Refused', 'Approved']

class CreditCardBalance(BaseModel):
    SK_ID_PREV: Optional[float] = Field(2562384)
    SK_ID_CURR: Optional[float] = Field(378907)
    MONTHS_BALANCE: Optional[float] = Field(-6)
    AMT_BALANCE: Optional[float] = Field(56.97)
    AMT_CREDIT_LIMIT_ACTUAL: Optional[float] = Field(135000)
    AMT_DRAWINGS_ATM_CURRENT: Optional[float] = Field(0.0)
    AMT_DRAWINGS_CURRENT: Optional[float] = Field(877.5)
    AMT_DRAWINGS_OTHER_CURRENT: Optional[float] = Field(0.0)
    AMT_DRAWINGS_POS_CURRENT: Optional[float] = Field(877.5)
    AMT_INST_MIN_REGULARITY: Optional[float] = Field(1700.325)
    AMT_PAYMENT_CURRENT: Optional[float] = Field(1800.0)
    AMT_PAYMENT_TOTAL_CURRENT: Optional[float] = Field(1800.0)
    AMT_RECEIVABLE_PRINCIPAL: Optional[float] = Field(0.0)
    AMT_RECIVABLE: Optional[float] = Field(0.0)
    AMT_TOTAL_RECEIVABLE: Optional[float] = Field(0.0)
    CNT_DRAWINGS_ATM_CURRENT: Optional[float] = Field(0.0)
    CNT_DRAWINGS_CURRENT: Optional[float] = Field(1)
    CNT_DRAWINGS_OTHER_CURRENT: Optional[float] = Field(0.0)
    CNT_DRAWINGS_POS_CURRENT: Optional[float] = Field(1.0)
    CNT_INSTALMENT_MATURE_CUM: Optional[float] = Field(35.0)
    NAME_CONTRACT_STATUS: str = "Active"
    SK_DPD: Optional[float] = Field(0)
    SK_DPD_DEF: Optional[float] = Field(0)
    UTILIZATION_RATE: Optional[str] = Field(alias="_UTILIZATION_RATE")

    @validator("UTILIZATION_RATE", always=True)
    def set_UTILIZATION_RATE(cls, v, values, **kwargs):
        if values.get("AMT_CREDIT_LIMIT_ACTUAL") == 0:
            return 0
        return values.get("AMT_BALANCE") / values.get("AMT_CREDIT_LIMIT_ACTUAL")
     
    @validator('NAME_CONTRACT_STATUS') 
    def validate_contract_status(cls, contract_status):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(contract_status, POSSIBLE_NAME_CONTRACT_STATUS)