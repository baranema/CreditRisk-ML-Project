from typing import Optional 
from pydantic import BaseModel, Field, validator
import app_validators

class ApplicationMerged(BaseModel):
    SK_ID_CURR: Optional[float] = Field(145315)
    inst_payments_NUM_INSTALMENT_NUMBER: Optional[float] = Field(23.796296296296298)
    prev_app_NFLAG_INSURED_ON_APPROVAL: Optional[float] = Field(0.0)
    REGION_RATING_CLIENT_W_CITY: Optional[float] = Field(1)
    bureau_DAYS_CREDIT_UPDATE: Optional[float] = Field(-776.6666666666666)
    bureau_AMT_ANNUITY: Optional[float] = Field(16875.0)
    bureau_AMT_CREDIT_SUM: Optional[float] = Field(592500.0)
    prev_app_DAYS_LAST_DUE_1ST_VERSION: Optional[float] = Field(182202.0)
    AMT_ANNUITY: Optional[float] = Field(28395.0)
    DAYS_EMPLOYED: Optional[float] = Field(-1732)
    HOUSETYPE_MODE: Optional[str] = Field("block of flats")
    DAYS_BIRTH: Optional[float] = Field(-11556)
    bureau_DAYS_CREDIT: Optional[float] = Field(-1422.6666666666667)
    credit_bal_AMT_BALANCE: Optional[float] = Field(311042.698125)
    LIVE_CITY_NOT_WORK_CITY: Optional[float] = Field(0)
    EXT_SOURCE_2: Optional[float] = Field(0.6592637367355662)
    ELEVATORS_AVG: Optional[float] = Field(0.32)
    FLAG_OWN_REALTY: Optional[float] = Field(1)
    EMERGENCYSTATE_MODE: Optional[float] = Field(0.0)
    bureau_CREDIT_DAY_OVERDUE: Optional[float] = Field(0.0)
    FLOORSMAX_AVG: Optional[float] = Field(0.6667)
    NAME_CONTRACT_TYPE: Optional[str] = Field("Cash loans")
    REGION_RATING_CLIENT: Optional[float] = Field(1)
    prev_app_DAYS_FIRST_DRAWING: Optional[float] = Field(182258.5)
    REG_CITY_NOT_LIVE_CITY: Optional[float] = Field(0)
    prev_app_CNT_PAYMENT: Optional[float] = Field(16.5)
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(-1020.0)
    NAME_HOUSING_TYPE: Optional[str] = Field("House / apartment")
    CODE_GENDER: Optional[str] = Field("M")
    HOUR_APPR_PROCESS_START: Optional[float] = Field(9)
    credit_bal_CNT_DRAWINGS_CURRENT: Optional[float] = Field(1.2916666666666667)
    FLAG_EMP_PHONE: Optional[float] = Field(1)
    bureau_AMT_CREDIT_SUM_DEBT: Optional[float] = Field(48276.0)
    FLAG_DOCUMENT_8: Optional[float] = Field(0)
    AMT_CREDIT: Optional[float] = Field(263686.5)
    prev_app_DAYS_DECISION: Optional[float] = Field(-648.8)
    EXT_SOURCE_1: Optional[float] = Field(0.2954345928984676)
    inst_payments_NUM_INSTALMENT_VERSION: Optional[float] = Field(0.09259259259259259)
    FLAG_PHONE: Optional[float] = Field(0)
    WEEKDAY_APPR_PROCESS_START: Optional[str] = Field("THURSDAY")
    bureau_AMT_CREDIT_SUM_OVERDUE: Optional[float] = Field(0.0)
    prev_app_HOUR_APPR_PROCESS_START: Optional[float] = Field(17.0)
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = Field(0.0)
    ORGANIZATION_TYPE: Optional[str] = Field("Business Entity Type 3")
    prev_app_AMT_DOWN_PAYMENT: Optional[float] = Field(15745.5)
    FLAG_DOCUMENT_3: Optional[float] = Field(1)
    FLAG_DOCUMENT_6: Optional[float] = Field(0)
    credit_bal_MONTHS_BALANCE: Optional[float] = Field(-12.5)
    AMT_INCOME_TOTAL: Optional[float] = Field(292500.0)
    NAME_EDUCATION_TYPE: Optional[str] = Field("Higher education")
    credit_bal_AMT_DRAWINGS_ATM_CURRENT: Optional[float] = Field(0.0)
    TOTALAREA_MODE: Optional[float] = Field(0.4936)
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(1.0)
    credit_bal_is_fraud: Optional[float] = Field(0.25)
    NAME_INCOME_TYPE: Optional[str] = Field("Commercial associate")
    EXT_SOURCE_3: Optional[float] = Field(0.7366226976503176)
    REG_CITY_NOT_WORK_CITY: Optional[float] = Field(0)
    credit_bal_CNT_DRAWINGS_ATM_CURRENT: Optional[float] = Field(0.0)
    FLAG_EMAIL: Optional[float] = Field(1)
    REGION_POPULATION_RELATIVE: Optional[float] = Field(0.04622)
    NAME_FAMILY_STATUS: Optional[str] = Field("Single / not married")
    OCCUPATION_TYPE: Optional[str] = Field("Managers")
    POS_SK_DPD_DEF: Optional[float] = Field(0.0)
    credit_bal_anomaly_score: Optional[float] = Field(0.014988913760043326)
    CNT_CHILDREN: Optional[float] = Field(0)
    credit_bal_AMT_DRAWINGS_CURRENT: Optional[float] = Field(36578.35125)
    prev_app_AMT_ANNUITY: Optional[float] = Field(54184.7475)
    prev_app_DAYS_FIRST_DUE: Optional[float] = Field(-846.0)
    NONLIVINGAPARTMENTS_AVG: Optional[float] = Field(0.0077)
    OWN_CAR_AGE: Optional[float] = Field(9.0)

    @validator('HOUSETYPE_MODE')
    def validate_housetype_mode(cls, housetype_mode):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(housetype_mode, ['block of flats', 'terraced house', 'specific housing'])
                   
    @validator('NAME_CONTRACT_TYPE')
    def validate_name_contract_type(cls, name_contract_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_contract_type, ['Cash loans', 'Revolving loans'])
                   
    @validator('NAME_HOUSING_TYPE')
    def validate_name_housing_type(cls, name_housing_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_housing_type, ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Co-op apartment', 'Office apartment'])
                   
    @validator('CODE_GENDER')
    def validate_code_gender(cls, code_gender):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(code_gender, ['M', 'F', 'XNA'])
                   
    @validator('WEEKDAY_APPR_PROCESS_START')
    def validate_weekday_appr_process_start(cls, weekday_appr_process_start):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(weekday_appr_process_start, ['THURSDAY', 'TUESDAY', 'FRIDAY', 'WEDNESDAY', 'SATURDAY', 'SUNDAY', 'MONDAY'])
                   
    @validator('ORGANIZATION_TYPE')
    def validate_organization_type(cls, organization_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(organization_type, ['Business Entity Type 3', 'Self-employed', 'XNA', 'Trade: type 3', 'Trade: type 7', 'Restaurant', 'Transport: type 4', 'Industry: type 11', 'Industry: type 9', 'Construction', 'Medicine', 'Other', 'Government', 'Bank', 'Kindergarten', 'Business Entity Type 1', 'Security', 'Business Entity Type 2', 'Industry: type 4', 'School', 'University', 'Trade: type 4', 'Industry: type 3', 'Industry: type 5', 'Trade: type 2', 'Hotel', 'Agriculture', 'Transport: type 2', 'Police', 'Housing', 'Industry: type 1', 'Transport: type 1', 'Culture', 'Military', 'Industry: type 7', 'Industry: type 12', 'Transport: type 3', 'Services', 'Cleaning', 'Electricity', 'Security Ministries', 'Industry: type 2', 'Religion', 'Postal', 'Industry: type 6', 'Advertising', 'Emergency', 'Trade: type 6', 'Trade: type 1', 'Insurance', 'Telecom', 'Legal Services', 'Mobile', 'Realtor', 'Trade: type 5', 'Industry: type 10', 'Industry: type 13', 'Industry: type 8'])
                   
    @validator('NAME_EDUCATION_TYPE')
    def validate_name_education_type(cls, name_education_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_education_type, ['Secondary / secondary special', 'Higher education', 'Lower secondary', 'Incomplete higher', 'Academic degree'])
                   
    @validator('NAME_INCOME_TYPE')
    def validate_name_income_type(cls, name_income_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_income_type, ['Commercial associate', 'Working', 'State servant', 'Pensioner', 'Student', 'Maternity leave', 'Businessman', 'Unemployed'])
                   
    @validator('NAME_FAMILY_STATUS')
    def validate_name_family_status(cls, name_family_status):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_family_status, ['Married', 'Single / not married', 'Widow', 'Separated', 'Civil marriage', 'Unknown'])
                   
    @validator('OCCUPATION_TYPE')
    def validate_occupation_type(cls, occupation_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(occupation_type, ['Sales staff', 'Managers', 'High skill tech staff', 'Laborers', 'Core staff', 'Accountants', 'Low-skill Laborers', 'Drivers', 'Medicine staff', 'Secretaries', 'Security staff', 'Cooking staff', 'Cleaning staff', 'IT staff', 'Private service staff', 'Waiters/barmen staff', 'HR staff', 'Realty agents'])
                   