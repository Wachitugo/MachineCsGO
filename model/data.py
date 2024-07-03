import pandas as pd

data_path = 'dataset/Anexo_ET_demo_round_traces.csv'
df = pd.read_csv(data_path, sep=';')


#Modificación de equipo por team id
for partida in df['MatchId'].unique():
  team1 = df[(df['RoundId']==1) & (df['InternalTeamId']==1) & (df['MatchId']==partida)]['Team'].unique()[0]
  team2 = df[(df['RoundId']==1) & (df['InternalTeamId']==2) & (df['MatchId']==partida)]['Team'].unique()[0]
  rondas = df[(df['MatchId']==partida)]['RoundId'].unique()
  for ronda in rondas:
    if ronda<16:
      df.loc[(df['RoundId']==ronda) & (df['InternalTeamId']==1) & (df['MatchId']==partida), 'Team'] = team1
      df.loc[(df['RoundId']==ronda) & (df['InternalTeamId']==2) & (df['MatchId']==partida), 'Team'] = team2
    elif ronda>=16:
      df.loc[(df['RoundId']==ronda) & (df['InternalTeamId']==1) & (df['MatchId']==partida), 'Team'] = team2
      df.loc[(df['RoundId']==ronda) & (df['InternalTeamId']==2) & (df['MatchId']==partida), 'Team'] = team1



#Cambio de datos: Cambiar de False4 en RoundWinner a False, agregar False a MatchWinner, ya que en contexto aplica
df.loc[29,'MatchWinner'] = 'False'
df.loc[29,'RoundWinner'] = 'False'

#Se eliminará columna Unnamed: 0, ya que solo contiene las ID de las filas
#Se eliminará AbnormalMatch, ya que contiene solamente valores False
df.drop(columns=['Unnamed: 0'], inplace=True)
df.drop(columns=['AbnormalMatch'], inplace=True)

#Eliminación de partidas que tengan menos de 16 rondas
for partida in df['MatchId'].unique():
  if df[df['MatchId']==partida]['RoundId'].unique().size<16:
    df.drop(df[df['MatchId']==partida].index, inplace=True)

#Eliminación de filas que corresponda tengan más de 31 rondas
for partida in df[df['RoundId']>31]['MatchId'].unique():
  df.drop(df[df['MatchId']==partida].index, inplace=True)


#Eliminación de partidas que tengan más o menos jugadores en un equipo
for partida in df['MatchId'].unique():
  cant_t1 = df[(df['InternalTeamId']==1) & (df['MatchId']==partida)]['Map'].count()
  cant_t2 = df[(df['InternalTeamId']==2) & (df['MatchId']==partida)]['Map'].count()
  if(cant_t1-cant_t2!=0):
    df.drop(df[df['MatchId']==partida].index, inplace=True)

#Dada la naturaleza de los datos, y dado que en cada columna hay datos str y bool,
#se cambiará ambas filas (match y round winner) a str

df['MatchWinner'] = df['MatchWinner'].astype(str)
df['RoundWinner'] = df['RoundWinner'].astype(str)

#Obtención de un valor visible para las columnas de armas
armas = ['PrimaryAssaultRifle', 'PrimarySniperRifle','PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol']
for tipo_arma in armas:
  df[tipo_arma] = (df[tipo_arma] * df['RoundStartingEquipmentValue']).astype(int)


#Transformación de filas booleanas a 0 y 1
df['MatchWinner'] = (df['MatchWinner']=="True").astype(int)
df['RoundWinner'] = (df['RoundWinner']=="True").astype(int)
df['Survived'] = (df['Survived']==True).astype(int)


#Consideración de algunas columnas para trabajar sobre ellas,
#además de cambio de nombre para mejor identificación
columnas = ['Mapa','Equipo','NumInterno','NumPartida','NumRonda','KillsRonda','DineroIndividual','ValorAR', 'ValorSR','ValorPesado', 'ValorSMG', 'ValorPistola','DineroEquipo','GranadasLetales','GranadasNoLetales','Sobrevive','GanaRonda','GanaPartida']
datos_considerar = df[['Map','Team','InternalTeamId','MatchId','RoundId','RoundKills','RoundStartingEquipmentValue','PrimaryAssaultRifle', 'PrimarySniperRifle','PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol','TeamStartingEquipmentValue','RLethalGrenadesThrown','RNonLethalGrenadesThrown','Survived','RoundWinner','MatchWinner']].copy()
datos_considerar.rename(columns=dict(zip(datos_considerar.columns, columnas)), inplace=True)



#Reemplazo de Mapa y Equipo por valores numéricos
datos_considerar.Mapa.replace({"de_inferno":1, "de_nuke":2, "de_mirage":3, "de_dust2":4}, inplace=True)
datos_considerar.Equipo.replace({"Terrorist":1, "CounterTerrorist":2}, inplace=True)


#Dataframe  con datos a considerar para el modelo
sin_bajos = pd.DataFrame()

sin_bajos['DineroEquipo'] = datos_considerar['DineroEquipo']
sin_bajos['Partida'] = datos_considerar['NumPartida']
sin_bajos['Ronda'] = datos_considerar['NumRonda']
sin_bajos['Equipo'] = datos_considerar['NumInterno']
sin_bajos['Granadas'] = (datos_considerar['GranadasLetales']+datos_considerar['GranadasNoLetales'])
sin_bajos['Kills'] = datos_considerar['KillsRonda']
sin_bajos['Ganada'] = datos_considerar['GanaPartida']
sin_bajos['GanaRonda'] = datos_considerar['GanaRonda']

sin_bajos_ag = sin_bajos.groupby(['Partida','Ronda', 'Equipo']).agg({
    'Partida':'first',
    'DineroEquipo':'first',
    'Ronda':'first',
    'Equipo':'first',
    'Granadas':'sum',
    'Kills':'sum',
    'GanaRonda':'first'
})

pickle_path = 'checkpoints/sin_bajos_ag.pkl'
sin_bajos_ag.to_pickle(pickle_path)