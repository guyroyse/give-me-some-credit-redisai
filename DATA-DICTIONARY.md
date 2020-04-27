# Notes on Give Me Some Credit

| Type    | Field                                | Values     | Description                                                                                                                                              |
|:------- |:------------------------------------ |:----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Target  | SeriousDlqin2yrs                     | Y/N        | Person experienced 90 days past due delinquency or worse                                                                                                 |
| Feature | RevolvingUtilizationOfUnsecuredLines | percentage | Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits |
| Feature | age                                  | integer    | Age of borrower in years                                                                                                                                 |
| Feature | NumberOfTime30-59DaysPastDueNotWorse | integer    | Number of times borrower has been 30-59 days past due but no worse in the last 2 years.                                                                  |
| Feature | DebtRatio                            | percentage | Monthly debt payments, alimony,living costs divided by monthy gross income                                                                               |
| Feature | MonthlyIncome                        | real       | Monthly income                                                                                                                                           |
| Feature | NumberOfOpenCreditLinesAndLoans      | integer    | Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)                                                     |
| Feature | NumberOfTimes90DaysLate              | integer    | Number of times borrower has been 90 days or more past due.                                                                                              |
| Feature | NumberRealEstateLoansOrLines         | integer    | Number of mortgage and real estate loans including home equity lines of credit                                                                           |
| Feature | NumberOfTime60-89DaysPastDueNotWorse | integer    | Number of times borrower has been 60-89 days past due but no worse in the last 2 years.                                                                  |
| Feature | NumberOfDependents                   | integer    | Number of dependents in family excluding themselves (spouse, children etc.)                                                                              |
