from typing import Optional 
from pydantic import BaseModel, validator, Field
import app_validators

class Application(BaseModel):
    SK_ID_CURR: Optional[float] = Field(100002)
    NAME_CONTRACT_TYPE: Optional[str] = Field("Cash loans")
    CODE_GENDER: Optional[str] = Field("M")
    FLAG_OWN_CAR: Optional[str] = Field("N")
    FLAG_OWN_REALTY: Optional[str] = Field("Y")
    CNT_CHILDREN: Optional[float] = Field(0)
    AMT_INCOME_TOTAL: Optional[float] = Field(202500.0)
    AMT_CREDIT: Optional[float] = Field(406597.5)
    AMT_ANNUITY: Optional[float] = Field(24700.5)
    AMT_GOODS_PRICE: Optional[float] = Field(351000.0)
    NAME_TYPE_SUITE: Optional[str] = Field("Unaccompanied")
    NAME_INCOME_TYPE: Optional[str] = Field("Working")
    NAME_EDUCATION_TYPE: Optional[str] = Field("Secondary / secondary special")
    NAME_FAMILY_STATUS: Optional[str] = Field("Single / not married")
    NAME_HOUSING_TYPE: Optional[str] = Field("House / apartment")
    REGION_POPULATION_RELATIVE: Optional[float] = Field(0.018801)
    DAYS_BIRTH: Optional[float] = Field(-9461)
    DAYS_EMPLOYED: Optional[float] = Field(-637)
    DAYS_REGISTRATION: Optional[float] = Field(-3648.0)
    DAYS_ID_PUBLISH: Optional[float] = Field(-2120)
    OWN_CAR_AGE: Optional[float] = Field(3)
    FLAG_MOBIL: Optional[float] = Field(1)
    FLAG_EMP_PHONE: Optional[float] = Field(1)
    FLAG_WORK_PHONE: Optional[float] = Field(0)
    FLAG_CONT_MOBILE: Optional[float] = Field(1)
    FLAG_PHONE: Optional[float] = Field(1)
    FLAG_EMAIL: Optional[float] = Field(0)
    OCCUPATION_TYPE: Optional[str] = Field("Laborers")
    CNT_FAM_MEMBERS: Optional[float] = Field(1.0)
    REGION_RATING_CLIENT: Optional[float] = Field(2)
    REGION_RATING_CLIENT_W_CITY: Optional[float] = Field(2)
    WEEKDAY_APPR_PROCESS_START: Optional[str] = Field("WEDNESDAY")
    HOUR_APPR_PROCESS_START: Optional[float] = Field(10)
    REG_REGION_NOT_LIVE_REGION: Optional[float] = Field(0)
    REG_REGION_NOT_WORK_REGION: Optional[float] = Field(0)
    LIVE_REGION_NOT_WORK_REGION: Optional[float] = Field(0)
    REG_CITY_NOT_LIVE_CITY: Optional[float] = Field(0)
    REG_CITY_NOT_WORK_CITY: Optional[float] = Field(0)
    LIVE_CITY_NOT_WORK_CITY: Optional[float] = Field(0)
    ORGANIZATION_TYPE: Optional[str] = Field("Business Entity Type 3")
    EXT_SOURCE_1: Optional[float] = Field(0.0830369673913225)
    EXT_SOURCE_2: Optional[float] = Field(0.2629485927471776)
    EXT_SOURCE_3: Optional[float] = Field(0.1393757800997895)
    APARTMENTS_AVG: Optional[float] = Field(0.0247)
    BASEMENTAREA_AVG: Optional[float] = Field(0.0369)
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = Field(0.9722)
    YEARS_BUILD_AVG: Optional[float] = Field(0.6192)
    COMMONAREA_AVG: Optional[float] = Field(0.0143)
    ELEVATORS_AVG: Optional[float] = Field(0.0)
    ENTRANCES_AVG: Optional[float] = Field(0.069)
    FLOORSMAX_AVG: Optional[float] = Field(0.0833)
    FLOORSMIN_AVG: Optional[float] = Field(0.125)
    LANDAREA_AVG: Optional[float] = Field(0.0369)
    LIVINGAPARTMENTS_AVG: Optional[float] = Field(0.0202)
    LIVINGAREA_AVG: Optional[float] = Field(0.019)
    NONLIVINGAPARTMENTS_AVG: Optional[float] = Field(0.0)
    NONLIVINGAREA_AVG: Optional[float] = Field(0.0)
    APARTMENTS_MODE: Optional[float] = Field(0.0252)
    BASEMENTAREA_MODE: Optional[float] = Field(0.0383)
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = Field(0.9722)
    YEARS_BUILD_MODE: Optional[float] = Field(0.6341)
    COMMONAREA_MODE: Optional[float] = Field(0.0144)
    ELEVATORS_MODE: Optional[float] = Field(0.0)
    ENTRANCES_MODE: Optional[float] = Field(0.069)
    FLOORSMAX_MODE: Optional[float] = Field(0.0833)
    FLOORSMIN_MODE: Optional[float] = Field(0.125)
    LANDAREA_MODE: Optional[float] = Field(0.0377)
    LIVINGAPARTMENTS_MODE: Optional[float] = Field(0.022)
    LIVINGAREA_MODE: Optional[float] = Field(0.0198)
    NONLIVINGAPARTMENTS_MODE: Optional[float] = Field(0.0)
    NONLIVINGAREA_MODE: Optional[float] = Field(0.0)
    APARTMENTS_MEDI: Optional[float] = Field(0.025)
    BASEMENTAREA_MEDI: Optional[float] = Field(0.0369)
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = Field(0.9722)
    YEARS_BUILD_MEDI: Optional[float] = Field(0.6243)
    COMMONAREA_MEDI: Optional[float] = Field(0.0144)
    ELEVATORS_MEDI: Optional[float] = Field(0.0)
    ENTRANCES_MEDI: Optional[float] = Field(0.069)
    FLOORSMAX_MEDI: Optional[float] = Field(0.0833)
    FLOORSMIN_MEDI: Optional[float] = Field(0.125)
    LANDAREA_MEDI: Optional[float] = Field(0.0375)
    LIVINGAPARTMENTS_MEDI: Optional[float] = Field(0.0205)
    LIVINGAREA_MEDI: Optional[float] = Field(0.0193)
    NONLIVINGAPARTMENTS_MEDI: Optional[float] = Field(0.0)
    NONLIVINGAREA_MEDI: Optional[float] = Field(0.0)
    FONDKAPREMONT_MODE: Optional[str] = Field("reg oper account")
    HOUSETYPE_MODE: Optional[str] = Field("block of flats")
    TOTALAREA_MODE: Optional[float] = Field(0.0149)
    WALLSMATERIAL_MODE: Optional[str] = Field("Stone, brick")
    EMERGENCYSTATE_MODE: Optional[str] = Field("No")
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(2.0)
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(2.0)
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(2.0)
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(2.0)
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(-1134.0)
    FLAG_DOCUMENT_2: Optional[float] = Field(0)
    FLAG_DOCUMENT_3: Optional[float] = Field(1)
    FLAG_DOCUMENT_4: Optional[float] = Field(0)
    FLAG_DOCUMENT_5: Optional[float] = Field(0)
    FLAG_DOCUMENT_6: Optional[float] = Field(0)
    FLAG_DOCUMENT_7: Optional[float] = Field(0)
    FLAG_DOCUMENT_8: Optional[float] = Field(0)
    FLAG_DOCUMENT_9: Optional[float] = Field(0)
    FLAG_DOCUMENT_10: Optional[float] = Field(0)
    FLAG_DOCUMENT_11: Optional[float] = Field(0)
    FLAG_DOCUMENT_12: Optional[float] = Field(0)
    FLAG_DOCUMENT_13: Optional[float] = Field(0)
    FLAG_DOCUMENT_14: Optional[float] = Field(0)
    FLAG_DOCUMENT_15: Optional[float] = Field(0)
    FLAG_DOCUMENT_16: Optional[float] = Field(0)
    FLAG_DOCUMENT_17: Optional[float] = Field(0)
    FLAG_DOCUMENT_18: Optional[float] = Field(0)
    FLAG_DOCUMENT_19: Optional[float] = Field(0)
    FLAG_DOCUMENT_20: Optional[float] = Field(0)
    FLAG_DOCUMENT_21: Optional[float] = Field(0)
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = Field(0.0)
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = Field(0.0)
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = Field(0.0)
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = Field(0.0)
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = Field(0.0)
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = Field(1.0)
    
    @validator('NAME_CONTRACT_TYPE')
    def validate_name_contract_type(cls, name_contract_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_contract_type, ['Cash loans', 'Revolving loans'])
                   
    @validator('CODE_GENDER')
    def validate_code_gender(cls, code_gender):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(code_gender, ['M', 'F'])
                   
    @validator('FLAG_OWN_CAR')
    def validate_flag_own_car(cls, flag_own_car):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(flag_own_car, ['N', 'Y'])
                   
    @validator('FLAG_OWN_REALTY')
    def validate_flag_own_realty(cls, flag_own_realty):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(flag_own_realty, ['Y', 'N'])
                   
    @validator('NAME_TYPE_SUITE')
    def validate_name_type_suite(cls, name_type_suite):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_type_suite, ['Unaccompanied', 'Family', 'Spouse, partner', 'Children', 'Other_A', 'Other_B', 'Group of people'])
                   
    @validator('NAME_INCOME_TYPE')
    def validate_name_income_type(cls, name_income_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_income_type, ['Working', 'State servant', 'Commercial associate', 'Pensioner', 'Unemployed', 'Student', 'Businessman', 'Maternity leave'])
                   
    @validator('NAME_EDUCATION_TYPE')
    def validate_name_education_type(cls, name_education_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_education_type, ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'])
                   
    @validator('NAME_FAMILY_STATUS')
    def validate_name_family_status(cls, name_family_status):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_family_status, ['Single / not married', 'Married', 'Civil marriage', 'Widow', 'Separated', 'Unknown'])
                   
    @validator('NAME_HOUSING_TYPE')
    def validate_name_housing_type(cls, name_housing_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(name_housing_type, ['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Office apartment', 'Co-op apartment'])
                   
    @validator('OCCUPATION_TYPE')
    def validate_occupation_type(cls, occupation_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(occupation_type, ['Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers', 'Sales staff', 'Cleaning staff', 'Cooking staff', 'Private service staff', 'Medicine staff', 'Security staff', 'High skill tech staff', 'Waiters/barmen staff', 'Low-skill Laborers', 'Realty agents', 'Secretaries', 'IT staff', 'HR staff'])
                   
    @validator('WEEKDAY_APPR_PROCESS_START')
    def validate_weekday_appr_process_start(cls, weekday_appr_process_start):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(weekday_appr_process_start, ['WEDNESDAY', 'MONDAY', 'THURSDAY', 'SUNDAY', 'SATURDAY', 'FRIDAY', 'TUESDAY'])
                   
    @validator('ORGANIZATION_TYPE')
    def validate_organization_type(cls, organization_type):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(organization_type, ['Business Entity Type 3', 'School', 'Government', 'Religion', 'Other', 'XNA', 'Electricity', 'Medicine', 'Business Entity Type 2', 'Self-employed', 'Transport: type 2', 'Construction', 'Housing', 'Kindergarten', 'Trade: type 7', 'Industry: type 11', 'Military', 'Services', 'Security Ministries', 'Transport: type 4', 'Industry: type 1', 'Emergency', 'Security', 'Trade: type 2', 'University', 'Transport: type 3', 'Police', 'Business Entity Type 1', 'Postal', 'Industry: type 4', 'Agriculture', 'Restaurant', 'Culture', 'Hotel', 'Industry: type 7', 'Trade: type 3', 'Industry: type 3', 'Bank', 'Industry: type 9', 'Insurance', 'Trade: type 6', 'Industry: type 2', 'Transport: type 1', 'Industry: type 12', 'Mobile', 'Trade: type 1', 'Industry: type 5', 'Industry: type 10', 'Legal Services', 'Advertising', 'Trade: type 5', 'Cleaning', 'Industry: type 13', 'Trade: type 4', 'Telecom', 'Industry: type 8', 'Realtor', 'Industry: type 6'])
                   
    @validator('FONDKAPREMONT_MODE')
    def validate_fondkapremont_mode(cls, fondkapremont_mode):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(fondkapremont_mode, ['reg oper account', 'org spec account', 'reg oper spec account'])
                   
    @validator('HOUSETYPE_MODE')
    def validate_housetype_mode(cls, housetype_mode):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(housetype_mode, ['block of flats', 'terraced house', 'specific housing'])
                   
    @validator('WALLSMATERIAL_MODE')
    def validate_wallsmaterial_mode(cls, wallsmaterial_mode):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(wallsmaterial_mode, ['Stone, brick', 'Block', 'Panel', 'Mixed', 'Wooden', 'Others', 'Monolithic'])
                   
    @validator('EMERGENCYSTATE_MODE')
    def validate_emergencystate_mode(cls, emergencystate_mode):
        return app_validators.CategoricalValsValidator.must_be_in_existing_values(emergencystate_mode, ['No', 'Yes'])
                   