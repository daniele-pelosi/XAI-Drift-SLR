bpic2015_1 = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name', # Will be also used as dynamic categorical column
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ["question",
                                    "monitoringResource",
                                    "org:resource"],
    'static_categorical_columns': ["case:Responsible_actor",
                                   'Aanleg (Uitvoeren werk of werkzaamheid)',
                                   'Bouw', 'Brandveilig gebruik (vergunning)',
                                   'Gebiedsbescherming',
                                   'Handelen in strijd met regels RO',
                                   'Inrit/Uitweg',
                                   'Kap',
                                   'Milieu (neutraal wijziging)',
                                   'Milieu (omgevingsvergunning beperkte milieutoets)',
                                   'Milieu (vergunning)',
                                   'Monument',
                                   'Reclame',
                                   'Sloop',
                                   'Brandveilig gebruik (melding)',
                                   'Milieu (melding)'],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ["case:SUMleges"]
}

bpic2015_2 = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ["question",
                                    "monitoringResource",
                                    "org:resource"],
    'static_categorical_columns': ["case:Responsible_actor",
                                   'Aanleg (Uitvoeren werk of werkzaamheid)',
                                   'Bouw',
                                   'Brandveilig gebruik (vergunning)',
                                   'Gebiedsbescherming',
                                   'Handelen in strijd met regels RO',
                                   'Inrit/Uitweg',
                                   'Kap',
                                   'Milieu (neutraal wijziging)',
                                   'Milieu (omgevingsvergunning beperkte milieutoets)',
                                   'Milieu (vergunning)',
                                   'Monument',
                                   'Reclame',
                                   'Sloop',
                                   'Brandveilig gebruik (melding)',
                                   'Milieu (melding)'
                                   ],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ["case:SUMleges",

                                 ]
}

bpic2015_3 = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ["question",
                                    "monitoringResource",
                                    "org:resource"],
    'static_categorical_columns': ["case:Responsible_actor",
                                   'Aanleg (Uitvoeren werk of werkzaamheid)',
                                   'Bouw',
                                   'Brandveilig gebruik (vergunning)',
                                   'Gebiedsbescherming',
                                   'Handelen in strijd met regels RO',
                                   'Inrit/Uitweg',
                                   'Kap',
                                   'Milieu (neutraal wijziging)',
                                   'Milieu (omgevingsvergunning beperkte milieutoets)',
                                   'Milieu (vergunning)',
                                   'Monument',
                                   'Reclame',
                                   'Sloop',
                                   'Brandveilig gebruik (melding)',
                                   'Milieu (melding)',
                                   'Flora en Fauna'
                                   ],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ["case:SUMleges",

                                 ]
}

bpic2015_4 = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ["question",
                                    "monitoringResource",
                                    "org:resource"],
    'static_categorical_columns': ["case:Responsible_actor",
                                   'Aanleg (Uitvoeren werk of werkzaamheid)',
                                   'Bouw',
                                   'Brandveilig gebruik (vergunning)',
                                   'Gebiedsbescherming',
                                   'Handelen in strijd met regels RO',
                                   'Inrit/Uitweg',
                                   'Kap',
                                   'Milieu (neutraal wijziging)',
                                   'Milieu (omgevingsvergunning beperkte milieutoets)',
                                   'Milieu (vergunning)',
                                   'Monument',
                                   'Reclame',
                                   'Sloop'
                                   ],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ["case:SUMleges",

                                 ]
}

bpic2015_5 = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ["question",
                                    "monitoringResource",
                                    "org:resource"],
    'static_categorical_columns': ["case:Responsible_actor",
                                   'Aanleg (Uitvoeren werk of werkzaamheid)',
                                   'Bouw', 'Brandveilig gebruik (vergunning)',
                                   'Gebiedsbescherming',
                                   'Handelen in strijd met regels RO',
                                   'Inrit/Uitweg',
                                   'Kap',
                                   'Milieu (neutraal wijziging)',
                                   'Milieu (omgevingsvergunning beperkte milieutoets)',
                                   'Milieu (vergunning)',
                                   'Monument',
                                   'Reclame',
                                   'Sloop',
                                   'Brandveilig gebruik (melding)',
                                   'Milieu (melding)',
                                   'Flora en Fauna'
                                   ],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ["case:SUMleges",

                                 ]
}

credit = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': [],
    'static_categorical_columns': [],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': []
}

helpdesk = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ['Resource',
                                    'customer',
                                    'product',
                                    'seriousness_2',
                                    'service_level',
                                    'service_type',
                                    'workgroup'],
    'static_categorical_columns': ['responsible_section', 'support_section'],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': []
}

hospital = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ['diagnosis',
                                    'caseType',
                                    'org:resource',
                                    'flagD',
                                    'state',
                                    'closeCode',
                                    'actRed',
                                    'flagC',
                                    'version'],
    'static_categorical_columns': ['speciality'],
    'dynamic_numerical_columns': ['msgCount'],
    'static_numerical_columns': []
}

bpic2012a = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ['org:resource'],
    'static_categorical_columns': [],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ['case:AMOUNT_REQ']
}

bpic2012o = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ['org:resource'],
    'static_categorical_columns': [],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ['case:AMOUNT_REQ']
}

bpic2012w = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ['org:resource'],
    'static_categorical_columns': [],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ['case:AMOUNT_REQ']
}

sepsis = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ['InfectionSuspected', 'org:group', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie',
                                    'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor',
                                    'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature', 'DiagnosticUrinaryCulture',
                                    'SIRSCritLeucos', 'Oligurie', 'DiagnosticLacticAcid', 'Diagnose', 'Hypoxie', 'DiagnosticUrinarySediment', 'DiagnosticECG'],
    'static_categorical_columns': [],
    'dynamic_numerical_columns': ['Age', 'CRP', 'LacticAcid', 'Leucocytes'],
    'static_numerical_columns': []
}

bpic2020_DomesticDeclarations = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ['org:role'],
    'static_categorical_columns': [],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ['case:Amount']}

bpic2020_InternationalDeclarations = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ['org:role'],
    'static_categorical_columns': ['case:BudgetNumber',
                                   'case:Permit BudgetNumber'],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ['case:Amount',
                                 'case:Permit RequestedBudget']}

bpic2020_PrepaidTravelCost = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ['org:role'],
    'static_categorical_columns': ['case:OrganizationalEntity',
                                   'case:Permit BudgetNumber',
                                   'case:Project'],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ['case:RequestedAmount',
                                 'case:Permit RequestedBudget']}

bpic2020_RequestForPayment = {
    'case_id_column': 'case:concept:name',
    'activity_column': 'concept:name',
    'timestamp_column': 'time:timestamp',
    'dynamic_categorical_columns': ['org:role'],
    'static_categorical_columns': ['case:Project',
                                   'case:OrganizationalEntity'],
    'dynamic_numerical_columns': [],
    'static_numerical_columns': ['case:RequestedAmount']}

