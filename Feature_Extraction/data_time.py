from datetime import datetime, timedelta

# def split_weeks(dates):
#     # Creazione di una lista di oggetti datetime a partire dalle date
#     date_objs = [datetime.strptime(date, '%d/%m/%Y') for date in dates]
#
#     # Ordinamento della lista di date in ordine crescente
#     date_objs.sort()
#
#     # Calcolo della data di inizio e fine della prima settimana
#     start_date = date_objs[0]
#     end_date = start_date + timedelta(days=7 - start_date.weekday())
#
#     # Creazione di una lista di indici divisi per settimane
#     indices_by_week = []
#     current_week_indices = []
#     for i, date_obj in enumerate(date_objs):
#         if date_obj <= end_date:
#             current_week_indices.append(i)
#         else:
#             indices_by_week.append(current_week_indices)
#             current_week_indices = [i]
#             start_date = end_date
#             end_date = start_date + timedelta(days=7)
#
#     # Aggiunta degli indici dell'ultima settimana alla lista
#     indices_by_week.append(current_week_indices)
#
#     return indices_by_week


def split_weeks(dates):
    date_objs = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    date_objs.sort()
    start_date = date_objs[0]
    end_date = date_objs[-1]

    num_weeks = ((end_date - start_date).days // 7) + 1
    indices_by_week = [[] for _ in range(num_weeks)]

    for i, date_obj in enumerate(date_objs):
        days_diff = (date_obj - start_date).days
        week_num = days_diff // 7
        indices_by_week[week_num].append(i)

    return indices_by_week

# dates = ['2022-01-01','2022-01-15','2022-01-13', '2022-01-12', '2022-01-14', '2022-01-07', '2022-01-09', '2022-01-11', '2022-01-05']
# indices_by_week = split_weeks(dates)
# prova=['carmelo','rosso','balocco','nutella','caramella','bicicletta','caneretta','rosso','prete']
# print(indices_by_week)
# for i in range(0,len(indices_by_week)):
#     indices=indices_by_week[i]
#     vettore_prova=[]
#     for i,index in enumerate(indices):
#         vettore_prova.append(prova[index])
#
# print(vettore_prova)


from datetime import datetime


def trimestral_indices(dates_list,m):
    # Converti le date in oggetti datetime
    dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in dates_list]

    # Crea un dizionario che associa a ogni trimestre (anno, trimestre) la lista degli indici delle date in quel trimestre
    trimestral_indices = {}
    for i, date in enumerate(dates):
        year = date.year
        trimester = (date.month - 1) // m + 1
        key = (year, trimester)
        if key not in trimestral_indices:
            trimestral_indices[key] = []
        trimestral_indices[key].append(i)

    # Restituisci la lista di liste degli indici dei trimestri, ordinati per anno e trimestre
    sorted_keys = sorted(trimestral_indices.keys())
    return [trimestral_indices[key] for key in sorted_keys]

