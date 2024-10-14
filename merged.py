import dash
from dash import html, dcc, Input, Output, State, Patch
import dash_bootstrap_components as dbc
from dash import dash_table
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_ag_grid as dag
import os
from plotly.subplots import make_subplots
import io
import base64
import openpyxl
from itertools import islice
import dash_ag_grid as dag
import plotly.express as px

pump_calc_df = pd.read_excel("PumpModels4.1.xlsx")
model_dropdown = [{'label': model, 'value': model} for model in set(pump_calc_df["Model Name "])]
pump_find_df = pd.read_excel("PumpModels4.1.xlsx")

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.UNITED, dbc.icons.BOOTSTRAP],suppress_callback_exceptions=True)
k_dict = {0:0.017,1:0.05,2:0.083,4:0.073,6:0.056,8:0.042,10:0.034,12:0.029,14:0.026}

def description_card():
  return html.Div(
      id="description-card",
      children=[
          html.H5("Pump Models",style={"font-weight":"bold"}),
          html.H3("Welcome to the Pump Calculation and Finder",style={"font-weight":"bold"}),
          html.Div(
              id="intro",
              children="Explore pump models, based on the range of pumping speed and pressure. Click on the model names from the dropdown to visualize pump curves.",
          ),
          html.Br()
      ]
  )

def generate_control_card():
  '''
  Control card for the website
  to take inputs and a submit button.
  The function creates an Input Group
  using Dash Comonents library's InputGroup() function and Input components
  from Dash Core Components.

  Parameters:
    None

  Returns:
    html.Div component for the control card
  '''
  return html.Div([
      dcc.RadioItems(
        id='radio-input',
        options=[
        {"label":("Find a Pump"),"value":"Pump"},
        {"label":("Find a Green Alternative Pump"),"value":"Pump Model"},
        {"label":("Pump Selection for Loadlock"),"value":"Loadlock"}
        ],
        value='Pump Model'
        ),
      html.Div(id='radio-output')
    ])

find_a_pump = 1

@app.callback([Output('radio-output','children')],
  Input('radio-input','value'))
def update_radio(value):
    global default_graph_models,find_a_pump
    if value == 'Pump':
       default_graph_models = []
       find_a_pump = 0
       return [html.Div([html.Div([
           html.Br(),
                            html.Div(
                                children=[
                                    html.Div(children=[
                                        dcc.Markdown("** Pumping speed m3/h **",style={"width":"200px"}),
                                        dcc.Input(
                                          id = {'type':'user-input','index':2},
                                          value='',style={"margin-left":"50px"}),
                                    ],style={"display":"flex"}),
                                    html.Br(),
                                    html.Div([dcc.Markdown("** Ultimate Pressure in mTorr **",style={"width":"200px"}),
                                        dcc.Input(
                                          id = {'type':'user-input','index':3},
                                          value='',style={"margin-left":"50px"}),
                                    ], style={"display":"flex"})
                                    ]
                                ,style={"backgroundColor":"dark gray"}
                            )
                        ]),
              html.Div([
                  html.Div(html.Button(id="submit-button", children="Submit", n_clicks=0),
                                     style={"padding": "10px"}),]),
        html.Br(),
        html.Div([
                html.Div(id='output-div'), # Placeholder element for Table output
                html.Br(),
                html.Br(),
              html.Div(id='graph-dropdown'), # Placeholder element for Graph Dropdown
              html.Br(),
              html.Div(id='graph-div', style={"align-items": "center"}), # Placeholder element for Graph object
              ], style={"align-items":"center","margin":"10px"})

        ])]
    elif value == 'Pump Model':
       default_graph_models = []
       find_a_pump = 1
       return [html.Div([html.Div([
                    html.Br(),
                      html.Div(children=[
                          dcc.Markdown("** Find a Green Alternative Pump **",style={"width":"200px"}),
                          dcc.Dropdown(
                              id = {'type':'user-input','index':1},
                              options=[{'label': model, 'value': model} for model in set(pump_calc_df["Model Name "]) if model],
                              multi=False,
                              value=[],
                              clearable=True,
                              style={"width": "250px","margin-left":"50px"},
                          )],style={"display":"flex"}
                      ),
                      html.Div(html.Button(id="submit-button", children="Submit", n_clicks=0),
                                     style={"padding": "10px"}),
                  ]),
        html.Br(),
        html.Div([
                html.Div(id='output-div'), # Placeholder element for Table output
                html.Br(),
                html.Br(),
              html.Div(id='graph-dropdown'), # Placeholder element for Graph Dropdown
              html.Br(),
              html.Div(id='graph-div', style={"align-items": "center"}), # Placeholder element for Graph object
              ], style={"align-items":"center","margin":"10px"})]
      )]
    else:
       return [html.Div([html.Br(),html.Div([dcc.Markdown("** Enter the target Pressure **",style={"width":"200px"}),
            dcc.Input(id='input-2', type='number', placeholder='Target Pressure',style={"margin-left":"50px","width":"200px"}),
            dcc.Dropdown(id='Pressure-Units',placeholder="Units",options=["Torr","mTorr","Bar","mBar"],style={"width":"200px"}),],style={'display': 'flex'}),
            
            html.Br(),
            html.Div([dcc.Markdown("""** Enter the Volume of the LoadLock (in L) **""",style={"width":"200px"}),
            dcc.Input(id='input-3', type='number', placeholder='Chamber Volume',style={"margin-left":"50px","width":"200px"}),],style={'display': 'flex'}),
            html.Br(),

            html.Div([dcc.Markdown("** Enter the Pipe dimensions **",style={"width":"200px"}),
            dash_table.DataTable(
                id='input-table',
                columns=[
                    {'name': 'Length (in cm)', 'id': 'length', 'type': 'numeric'},
                    {'name': 'Diameter (in cm)', 'id': 'diameter', 'type': 'numeric'},
                ],
                data=[{'length': '', 'diameter': ''}],
                editable=True,
                row_deletable=True,
                style_cell={'width': '200px', 'textAlign': 'center'}
            ,style_table={"margin-left":"50px"})],style={"display":"flex"}),
            html.Br(),
            html.Button('Add Row', id='add-row-button', n_clicks=0,style={"margin-left":"250px","width":"120px","fontWeight":"bold","textAlign":"center"}),
            html.Br(),
            html.Br(),
            html.Div([dcc.Markdown("** Enter the Bend dimensions **",style={"width":"200px"}),
            dash_table.DataTable(
                id='input-table-2',
                columns=[
                    {'name': 'Diameter (in cm)', 'id': 'diameter', 'type': 'numeric'},
                ],
                data=[{'diameter': ''}],
                editable=True,
                row_deletable=True,
                style_cell={'width': '200px', 'textAlign': 'center'}
            ,style_table={"margin-left":"50px"})],style={"display":"flex"}),
            html.Br(),
            html.Button('Add Row', id='add-row-button-2', n_clicks=0,style={"margin-left":"250px","width":"120px","fontWeight":"bold","textAlign":"center"}),
            html.Br(),
            # html.Br(),
            # html.B("Enter or Select Pumping Speed"),
            # html.Br(),
            # html.Div([dcc.Dropdown(id='pumping-speed-1',options=df['Pumping speed m3/hr '].unique(),placeholder='Select Speed',style={"width":"250px"}),html.Br(),
            #           dcc.Input(id='pumping-speed-2',type='number',placeholder='Enter Pumping Speed',style={"width":"250px"})]),
            # html.Br(),
            # html.B("Enter the Flow Rate in Torr-L/s"),
            # html.Br(),
            # dcc.Input(id="flow-rate",type="number",placeholder="Flow Rate"),
            html.Br(),
            html.Div([dcc.Markdown("** Enter the Pump Down time in secs **",style={"width":"200px"}),
            dcc.Input(id='input-1', type='number', placeholder='Pump Down Time',style={"margin-left":"50px","width":"200px"})],style={"display":"flex"}),
            html.Br(),
            html.Div([dcc.Markdown("** Select a percentage of Effective Pumping speed for Data Filtering **"),
            
            dcc.Slider(0, 100, 10,
               value=30,
               id='S_eff'
            )]),
            html.Br(),
            html.Div([
                dcc.Markdown("** Excel Template Download and Upload **",style={"width":"200px"}),
            dbc.Button("Download Template", id="download-button", color="primary",style={"margin-left":"50px","width":"200px"}),
            dcc.Download(id="download-template"),  ],style={"display":"flex"}
            ),

            html.Br(),
            dcc.Markdown("""**Use the template file and fill the details of your configuration. You can add more rows.
                         Please unblock safety measures that blocks the macro in the excel file by going to the properties of the file.
                         Enable micros in the excel sheet.
                         While uploading make sure the file has only a single sheet in the workbook and it 
                         is an .xlsm file.**"""),
            html.Br(),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False, 
                accept=".xlsm"  
            ),
                
            html.Div(id='output-data-upload'),
            
            html.Br(),
            html.Button("Submit",id='submit-button-find',n_clicks=0),
            html.Div(id='output-container',style={"padding":"10px"}),
            html.Div(id='output-container-2',style={"padding":"10px"})
        ])]

###############################################################################################################################################################################################################
################################################################################################################################################################################################################


# Pump Finder functions

@app.callback(
    Output("download-template", "data"),
    [Input("download-button", "n_clicks")],
    prevent_initial_call=True
)
def download_template(n_clicks):
    if n_clicks:
        return dcc.send_file("template.xlsm")

@app.callback(
    [Output('output-data-upload', 'children',allow_duplicate=True),
    Output('input-table', 'data',allow_duplicate=True),
    Output('input-table-2','data',allow_duplicate=True)],
    [Input('upload-data', 'contents')],
    State('upload-data', 'filename'),
    State('input-table','data'),
    State('input-table-2','data'),
    State('input-table','columns'),
    State('input-table-2','columns'),prevent_initial_call=True
)
def handle_uploaded_file(contents, filename,rows_1,rows_2,cols1,cols2):
    if contents is not None:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if filename.endswith('.xlsm'):
            try:
                df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')  # Pandas does not directly support xlsm read, so this is just for illustration
                data = df.values
                
                
                table_1 = []
                table_2 = []
                for row in data:
                    if "bend" in row[1].lower():
                        table_2.append((row[2]))
                    else:
                        table_1.append([(row[2]),(row[4])])
                    
                rows_1=[]
                rows_2=[]
                for val in table_2:rows_2.append({'diameter':val})
                for a,b in table_1:rows_1.append({'length':a,'diameter':b})

                return [html.Div([
                    html.H5(f"Uploaded file: {filename}"),
                    html.Hr(),
                    html.H6("File content preview:"),
                    dcc.Markdown(df.head().to_markdown())
                ]),rows_1,rows_2]
            except Exception as e:
                return html.Div([
                    'There was an error processing the file.',
                    html.Br(),
                    f'Error: {e}'
                ],rows_1,rows_2)
        else:
            return [html.Div([
                'Please upload a valid .xlsm file.'
            ]),rows_1,rows_2]


@app.callback(
    Output('input-table', 'data',allow_duplicate=True),
    Input('add-row-button', 'n_clicks'),
    State('input-table', 'data'),
    State('input-table', 'columns'),prevent_initial_call=True
)
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        print(rows,columns)
        rows.append({c['id']: '' for c in columns})
    return rows

@app.callback(
    Output('input-table-2', 'data',allow_duplicate=True),
    Input('add-row-button-2', 'n_clicks'),
    State('input-table-2', 'data'),
    State('input-table-2', 'columns'),prevent_initial_call=True
)
def add_row_2(n_clicks, rows, columns):
    if n_clicks > 0:
        print(rows,columns)
        rows.append({c['id']: '' for c in columns})
    return rows

@app.callback(
    Output('output-container-2', 'children', allow_duplicate=True),
    Input('submit-button-find', 'n_clicks'),
    Input('input-1', 'value'),
    Input('input-2', 'value'),
    Input('input-3', 'value'),
    # State('pumping-speed-2','value'),
    State('input-table', 'data'),
    State('input-table-2','data'),
    # State('flow-rate','value'),
    State("Pressure-Units","value"),
    State("S_eff",'value'),
    prevent_initial_call=True
)
def calculate_downtimes(n_clicks, input1, input2, input3, table_data,table_data_2,pressure_units,slider):
    if n_clicks > 0:
        tp = input1
        pressure = input2
        if pressure_units == "mTorr":
            pressure = pressure*0.001
        elif pressure_units == 'Bar':
            pressure = pressure*760
        elif pressure_units == "mBar":
            pressure = pressure*0.76
        volume = input3
        C = 0
        mu = 0.0001782
        p_ = round((pressure+760)/2,3)
        global table_values
        global table_values_2
        table_values = []
        table_values_2 = []
        PI = 3.14
        
        
        for row in table_data:
            table_values.append([row['length'],row['diameter']])
        
        for row in table_data_2:
            table_values_2.append(row['diameter'])

        outputs=[]
        cs = []
        for l,d in table_values:
            l,d = round(l,4),round(d,4)
            c = 0.0327*(d**4/(mu*l))*p_
            # outputs.append(html.P(f"Conductance calculated for pipe with length {l}cm and diameter {d} cm = (({0.0327}*{d}^4)/({mu}*{l}))*{p_} = {c} L/s"))
            
            c = round(c,4)
            cs.append(c)
            C += 1/(c)
            # outputs.append(html.P(f"Inverse of Conductance 1/c = {1/c}"))
            # outputs.append(html.Br())
        
        constant_factor = (PI*1000*133.3*p_) / (128*mu*0.1)
        for d in table_values_2:
            K = k_dict[1] # r==d
            d = round(d,4)
            c = constant_factor*K*((d*0.01)**3)
            # outputs.append(html.P(f"Conductance calculated for bend with diameter {d}cm = (({PI}*{1000}*{133.3}*{p_})/({128}*{mu}*{0.1}))*{K}*(({d}*{0.01})^3) = {c} L/s"))
            # outputs.append(html.Br())
            c = round(c,4)
            cs.append(c)
            # outputs.append(html.P(f"Inverse of Conductance 1/c = {1/c}"))
            C += 1/(c)
        
        
        # outputs.append(html.P(f"Inverse Conductance values addition: 1/C = {" + ".join([str(1/c) for c in cs])}"))
        C = 1/C
        C = round(C,4)
        # outputs.append(html.B(f"Final Conductance Calculated C = {C} L/s"))
        global filtered_df
        filtered_df  = pump_find_df
        # try:
        #     filtered_df = df[(df["Pumping speed m3/hr "] >= pump_speed*0.9) & (df["Pumping speed m3/hr "] <= (pump_speed*1.1))]
        # except:
        #     pass
        filtered_df.round(2)
        cols = pump_find_df.columns
        
        # if flow_rate:
        #     Q = []
        #     steady_state_df = filtered_df[filtered_df["Ult pressure(mTorr) "] <= pressure]
        #     for So in steady_state_df[cols[2]].unique():
        #         So = float(So)
        #         S_eff = (So*C)/(So+C)
        #         Q.append([So,round(S_eff*pressure,4)])
        #     Q.sort(key=lambda x:abs(flow_rate-x[1]))
        #     Q_table = pd.DataFrame(Q,columns=["Pumping Speed","Flow Rate"])
        #     outputs.append(html.Br())
        #     outputs.append(html.B(f"Flow Rates table :"))
        #     outputs.append(html.Br())
        #     columns = [{'headerName': i, 'field': i} for i in Q_table.columns]
        #     data = Q_table.to_dict('records')
        #     grid = dag.AgGrid(
        #         id='flow-rate-div',
        #         columnDefs=columns,
        #         rowData=data,
        #         columnSize="sizeToFit",
        #         defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 100},
        #         dashGridOptions={"pagination": True, "paginationPageSize": 8, "domLayout": "autoHeight",'rowSelection': "single"},
        #         style={"height": 500, "width": "100%"},
        #         className="ag-theme-alpine"
        #     )
        #     outputs.append(html.Div(id="flow-rate-div", children=[grid],style={"padding":"10px"}))
            
        #     outputs.append(html.Div(id='Flow-Rate-Output'))
        
        filtered_df = filtered_df[filtered_df["Ult pressure(mTorr) "] <= pressure]
        # filtered_df["Effective Pumping Speed (L/s)"] = 0.0
        # for So in filtered_df["Pumping speed m3/hr "].unique():
        #     So_org = So
        #     So = float(So)*0.277778
        #     S_eff = (So*C)/(So+C)
        #     S_eff = round(S_eff,4)
        #     filtered_df.loc[filtered_df["Pumping speed m3/hr "] == So_org,"Effective Pumping Speed (L/s)"] = round(S_eff,2)
        # outputs.append(html.Br())
        target_S_eff = (volume/tp)*math.log(760/pressure)
        target_S_eff = round(target_S_eff,4)
        # outputs.append(html.P(f"Effective Speed calculation = ({volume}/{tp}) * log10(({760})/{pressure}) = {target_S_eff}"))
        # outputs.append(html.B(f"Effective Speed S_eff = {target_S_eff} L/S"))
        outputs.append(html.Div([
                # dcc.Markdown("Click here to take a look at the entire database",style={"width":"200px"}),
                dbc.Button("View Database", id="database-button", color="info",style={"width":"150"},n_clicks=0),
                html.Br(),
                html.Div(id='Database-Output')]
            ))
        outputs.append(html.Br())
        outputs.append(html.B(f"Table :"))
        outputs.append(html.Br())
        target_S = round((target_S_eff*C)/(C-target_S_eff),4)*3.6
        # outputs.append(html.P(f"Pumping Speed for Comparision = ({target_S_eff}*{C})/({C}-{target_S_eff}) = {target_S} m3/hr"))
        slider = slider/100 #put a slider
        filtered_df = filtered_df[(filtered_df["Pumping speed m3/hr "] >= (1-slider)*target_S) & (filtered_df["Pumping speed m3/hr "] <= (1+slider)*target_S)]
        filtered_df[['Total Equivalent Energy','Heat Balance','N2 kWh/year',"DE KWh/ year ",'PCW KWh/year ']] = filtered_df[['Total Equivalent Energy','Heat Balance','N2 kWh/year',"DE KWh/ year ",'PCW KWh/year ']].astype(int)
        filtered_df["Pump_DownTimes"] = [calculate_downtimes_pipes(model,pressure,volume) for model in filtered_df["Model Name "]]
        filtered_df = filtered_df.sort_values("Total Equivalent Energy")
        # cols = filtered_df.columns[0:2].to_list() + ["Pump_DownTimes"] + filtered_df.columns[2:-1].to_list()
        # print(target_S,pump_find_df['Pumping speed m3/hr '].values)
        cols = ["Supplier ","Model Name ","Pumping speed m3/hr ","Ult pressure(mTorr) ","Pump_DownTimes","Total Equivalent Energy","Inlet ","exhaust ","PCWmin lpm","Power at ultimate KW ","Heat Balance","N2 kWh/year","DE KWh/ year ","PCW KWh/year "]
        columns = [{"field":col} for col in cols]
        filtered_df = filtered_df[cols]
        data = filtered_df.to_dict('records')
        grid = dag.AgGrid(
            columnDefs=columns,
            rowData=data,
            columnSize="sizeToFit",
            defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 100},
            style={"height": None, "width": "100%"},
            dashGridOptions={
                        "pagination": True,
                        "paginationPageSize": 8,
                        "domLayout": "autoHeight",
                        'rowSelection': "single",
                        'getRowStyle': {
                                    'styleConditions': [
                                        {'condition': "params.rowIndex === 0",
                                                                'style': {
                                                                    'backgroundColor': 'green',
                                                                    'color': 'white'
                                                                }}
                       ]},
                    },
            className="ag-theme-alpine"
        )
        outputs.append(html.Div(id="table-div", children=[grid]))
        if filtered_df.empty or filtered_df["Pump_DownTimes"].iloc[0]!=tp:
            outputs.append(html.B("Note: If the pump down time provided in the input does not align with the values in the table, you may need to adjust the foreline dimensions to achieve a closer match. Changing the foreline dimensions will affect the overall system, which can help bring the calculated pump down time closer to your desired value.",style={"text-align":"center",'font-size':'15px'}))
        outputs.append(html.Br())
        outputs.append(html.Br())
        outputs.append(html.Div(id="Pumping-Speed-Output"))
        return outputs

    else:
        return dash.no_update


@app.callback(
        Output('Database-Output','children'),
        Input('database-button','n_clicks'),
        State('input-2', 'value'),
    State('input-3', 'value'),
)
def database_View(n_clicks,pressure,volume):
    if n_clicks:
        lst = [(calculate_downtimes_pipes(model,pressure,volume)) for model in pump_find_df["Model Name "]]
        pump_find_df["Pump_DownTimes"] = [round(val,4) if val else 0 for val in lst]
        
        cols = ["Supplier ","Model Name ","Pumping speed m3/hr ","Ult pressure(mTorr) ","Pump_DownTimes","Total Equivalent Energy","Inlet ","exhaust ","PCWmin lpm","Power at ultimate KW ","Heat Balance","N2 kWh/year","DE KWh/ year ","PCW KWh/year "]
        columns = [{"field":col} for col in cols]
        data = pump_find_df.to_dict('records')
        
        grid = dag.AgGrid(
            id='entire-table-div',
            columnDefs=columns,
            rowData=data,
            columnSize="sizeToFit",
            defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 100},
            style={"height": 500, "width": "100%"},
            dashGridOptions={
                        "pagination": True,
                        "paginationPageSize": 8,
                        "domLayout": "autoHeight",
                        'rowSelection': "single",
                    },
            className="ag-theme-alpine"
        )
        
        return [html.Div(id="entire-table-div", children=[grid],style={"padding":"10px"})]
    else:
        return dash.no_update


def extrapolate(pressures,speeds,initial_pressure, target_pressure):
    dt = DecisionTreeRegressor()
    X,y = pressures,speeds
    x = []
    for i in range(len(X)):
        x.append([np.log10(X[i])])
    x = np.array(x)
    x_train,x_test,y_train,y_test = train_test_split(x,np.array(y),test_size=0.1,shuffle=True,random_state=42)
    dt = DecisionTreeRegressor()
    dt.fit(x_train.reshape(-1,1),y_train)
    new_pressure_set = np.arange(target_pressure,initial_pressure,30)
    new_pressure_mod = np.array([np.log10(v) for v in new_pressure_set])
    new_speeds = dt.predict(new_pressure_mod.reshape(-1,1))
    return new_pressure_set[::-1],new_speeds[::-1]

def calculate_downtimes_pipes(model_name,target_pressure,volume):
    '''
    model_name : Model Name
    target_pressure : Input Target Pressure
    volume : chamber volume
    '''
    filename = f"{model_name}.csv"
    location = "Phase2_graphdata"
    if filename in os.listdir(location):
        pressures,speeds = extract(filename,location,target_pressure)
        pressures,speeds = extrapolate(pressures,speeds,760,target_pressure)
        # pressures = list(pressures) # [760] + list(pressures) + [target_pressure]
        # speeds =  list(speeds) #[-1] + list(speeds) + [selected_rows[0]["Pumping speed m3/hr "]]
        speeds = [speed*0.277777778 for speed in speeds]
        actual_pressures = list(pressures) # + [760] + [target_pressure]
        pressures = [(actual_pressures[i]+actual_pressures[i-1])/2 for i in range(1,len(pressures))]
        actual_speeds = list(speeds)
        
        new_speeds = []
        for i in range(1,len(actual_speeds)):
            new_speeds.append(10**(np.log10(actual_speeds[i-1]) + (((np.log10(pressures[i-1]/actual_pressures[i-1]))*(np.log10(actual_speeds[i]/actual_speeds[i-1])))/(np.log10(actual_pressures[i]/actual_pressures[i-1])))))
        
        speeds = new_speeds
        pfs = []
        conductances = []
        mu = 0.0001782
        PI = 3.14
        effective_speeds = []
        pump_down_times = []
        Q = []
        n = len(pressures)
        for i in range(n):
            v=volume
            So = speeds[i]
            
            p_ = (pressures[i])
            C = 0
            for l,d in table_values:
                l,d = round(l,4),round(d,4)
                c = 0.0327*(d**4/(mu*l))*p_
                c = round(c,4)
                v += round(PI*(d/2)**2*l,4) * 10**(-3)
                C += 1/(c)
            
            constant_factor = (PI*1000*133.3*p_) / (128*mu*0.1)
            for d in table_values_2:
                K = k_dict[1]
                d = round(d,4)
                c = constant_factor*K*((d*0.01)**3)
                c = round(c,4)
                C += 1/(c)

            C = 1/C
            conductances.append(C)
            S_eff = (So*C)/(So+C)
            effective_speeds.append(S_eff)
            if i!=n-1:
                time = round(((v/S_eff)*math.log(pressures[i]/pressures[i+1])),4)
                pump_down_times.append(time)
            else:
                pump_down_times.append(round(((v/S_eff)*math.log(pressures[i]/target_pressure)),4))
            pf = (S_eff/(C+S_eff))*760
            pfs.append(pf)
            Q.append(pressures[i]*S_eff)
            
        conductances = list(map(lambda x:round(x,2), conductances))
        effective_speeds = list(map(lambda x:round(x,2), effective_speeds))
        pump_down_times = list(map(lambda x:round(x,2), pump_down_times))
        speeds = list(map(lambda x:round(x,2), speeds))
        pressures = list(map(lambda x:round(x,2), pressures))
        Q = list(map(lambda x:round(x,2), Q))
        pfs = list(map(lambda x:round(x,2), pfs))
        return round(sum(pump_down_times),4)

def extract(filename,location,target_pressure):
    filename = os.path.join(os.path.join(os.getcwd(),location),filename)
    values = [row.strip("ï»¿\n").split(",") for row in open(filename).readlines()]
    values = [[float(v),float(w)] for v,w in values]
    values = [v for v in values if (target_pressure<=v[0]<=760)]
    values.sort(key=lambda x:x[0],reverse=True)
    return list(zip(*values))

@app.callback(
    Output('Pumping-Speed-Output', 'children'),
    Input('table-div', 'selectedRows'),
    State('input-2','value'),
    State('input-3','value')
)
def display_selected_row(selected_rows,target_pressure,volume):
    if selected_rows:
        selected_row = pump_find_df[pump_find_df["Pumping speed m3/hr "] == selected_rows[0]["Pumping speed m3/hr "]].sort_values("Total Equivalent Energy")
        outputs = []
        model_name = selected_row["Model Name "].values[0]
        filename = f"{model_name}.csv"
        location = "Phase2_graphdata"
        if filename in os.listdir(location):
            pressures,speeds = extract(filename,location,target_pressure)
            pressures,speeds = extrapolate(pressures,speeds,760,target_pressure)
            # pressures = list(pressures) # [760] + list(pressures) + [target_pressure]
            # speeds =  list(speeds) #[-1] + list(speeds) + [selected_rows[0]["Pumping speed m3/hr "]]
            speeds = [speed*0.277777778 for speed in speeds]
            actual_pressures = list(pressures) # + [760] + [target_pressure]
            pressures = [(actual_pressures[i]+actual_pressures[i-1])/2 for i in range(1,len(pressures))]
            actual_speeds = list(speeds)
            
            new_speeds = []
            for i in range(1,len(actual_speeds)):
                new_speeds.append(10**(np.log10(actual_speeds[i-1]) + (((np.log10(pressures[i-1]/actual_pressures[i-1]))*(np.log10(actual_speeds[i]/actual_speeds[i-1])))/(np.log10(actual_pressures[i]/actual_pressures[i-1])))))
            
            speeds = new_speeds
            pfs = []
            conductances = []
            mu = 0.0001782
            PI = 3.14
            effective_speeds = []
            pump_down_times = []
            Q = []
            n = len(pressures)
            for i in range(n):
                v=volume
                So = speeds[i]
                # outputs.append(html.Br())
                # outputs.append(html.P(f"Calculating with speed : {So}"))
                
                p_ = (pressures[i])
                C = 0
                # outputs.append(html.P(f"Average Pressure p_ = {p_}"))
                for l,d in table_values:
                    l,d = round(l,4),round(d,4)
                    c = 0.0327*(d**4/(mu*l))*p_
                    c = round(c,4)
                    v += round(PI*(d/2)**2*l,4) * 10**(-3)
                    C += 1/(c)
                    # outputs.append(html.P(f"Conductance for length {l}, and diameter {d} = {0.0327}*({d}^4/{mu}*{l})*{p_} = {c}"))
                
                constant_factor = (PI*1000*133.3*p_) / (128*mu*0.1)
                # outputs.append(html.P(f"Constant factor in calculation ({PI}*Average_Pressure*1000)/(128*mu*0.1) = {constant_factor}"))
                for d in table_values_2:
                    K = k_dict[1]
                    d = round(d,4)
                    c = constant_factor*K*((d*0.01)**3)
                    c = round(c,4)
                    C += 1/(c)
                    # outputs.append(html.P(f"Conductance calculated for Bend with diameter {d} = {constant_factor} * {K} * ({d}*0.01)^3  = {c}"))

                C = 1/C
                # outputs.append(html.P(f"Overall Conductance for this speed = {C}"))
                conductances.append(C)
                S_eff = (So*C)/(So+C)
                # outputs.append(html.P(f"Effective Pumping Speed S_eff = {So}*{C}/({So}+{C}) = {S_eff}"))
                effective_speeds.append(S_eff)
                if i!=n-1:
                    time = round(((v/S_eff)*math.log(pressures[i]/pressures[i+1])),4)
                    # outputs.append(html.P(f"Pump Down Time t = ({v}/{S_eff})*log({pressures[i]}/{pressures[i+1]}) = {time}"))
                    pump_down_times.append(time)
                else:
                    time = round(((v/S_eff)*math.log(pressures[i]/target_pressure)),4)
                    pump_down_times.append(time)
                    # outputs.append(html.P(f"Pump Down Time t = ({v}/{S_eff})*log({pressures[i]}/{target_pressure}) = {time}"))
                pf = (S_eff/(C+S_eff))*760
                pfs.append(pf)
                Q.append(pressures[i]*S_eff)
            
            conductances = list(map(lambda x:round(x,2), conductances))
            effective_speeds = list(map(lambda x:round(x,2), effective_speeds))
            pump_down_times = list(map(lambda x:round(x,2), pump_down_times))
            speeds = list(map(lambda x:round(x,2), speeds))
            pressures = list(map(lambda x:round(x,2), pressures))
            Q = list(map(lambda x:round(x,2), Q))
            pfs = list(map(lambda x:round(x,2), pfs))
            # another_table = pd.DataFrame({"Pressure (Torr)":pressures,"Pumping speed L/s ":speeds,"Conductance L/s":conductances,"Effective Speed L/s":effective_speeds,"Pump Down Time (s)":pump_down_times})
            # columns = [{'headerName': i, 'field': i} for i in another_table.columns]
            # data = another_table.to_dict('records')
            # grid = dag.AgGrid(
            #     id='flow-rate-div',
            #     columnDefs=columns,
            #     rowData=data,
            #     columnSize="sizeToFit",
            #     defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 100},
            #     dashGridOptions={"pagination": True, "paginationPageSize": 8, "domLayout": "autoHeight",'rowSelection': "single"},
            #     style={"height": 500, "width": "100%"},
            #     className="ag-theme-alpine"
            # )
            # outputs.append(html.Div(id="individual-pressures-div", children=[grid],style={"padding":"10px"}))
            
            # outputs.append(html.Br())
            # outputs.append(html.B(f"Total Pump Down Time t = {round(sum(pump_down_times),4)} s"))
            # outputs.append(html.Br())
        
            fig = make_subplots(specs=[[{"secondary_y": False}]]) # Canvas for plots
            fig.add_trace(go.Scatter(y=pressures[1:-1], x=pump_down_times,name="Time vs Pressure"))
                
            fig['layout'].update(height=600,
                                    width=800,
                                    title='Pump Down Curve',
                                    )
            fig.update_yaxes(title_text="Pressure Torr")
            fig.update_xaxes(title_text="Pump Down Times")

            outputs.extend([
            # html.Br(),
            dcc.Graph(figure=fig),
            # html.Br(),
            # html.Br()
            ])
        return html.Div(outputs)
    else:
        return html.P(html.B('Select any row to display Models and Graphs'))


@app.callback(
    Output('Flow-Rate-Output', 'children'),
    Input('flow-rate-div', 'selectedRows'),
    Input('input-2', 'value'),
)
def display_selected_row_2(selected_rows,pressure):
    if selected_rows:

        selected_row = pump_find_df[pump_find_df["Pumping speed m3/hr "] == selected_rows[0]["Pumping speed m3/hr "]].sort_values("Total Equivalent Energy")
        selected_row = selected_row[selected_row["Ult pressure(mTorr) "] <= pressure]
        return html.Div([
                dag.AgGrid(
                    rowData=selected_row.to_dict('records'),
                    columnDefs=[{'headerName': i, 'field': i} for i in selected_row.columns],
                    columnSize="sizeToFit",
                    defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 100},
                    dashGridOptions={
                        "pagination": True,
                        "paginationPageSize": 8,
                        "domLayout": "autoHeight",
                        'rowSelection': "single",
                        'getRowStyle': {
                                    'styleConditions': [
                                        {'condition': "params.rowIndex === 0",
                                                                'style': {
                                                                    'backgroundColor': 'green',
                                                                    'color': 'white'
                                                                }}
                       ]}
                    },
                    style={"height": 500, "width": "100%"},
                    className="ag-theme-alpine"
                ),
        html.Br(),
        html.B("Select model(s) for the Pressure vs Time graph (Not Configured yet)"),
        html.Br(),
        dcc.Dropdown(id="graph-dropdown-2",options=selected_row["Model Name "].to_list(),multi=True,value=[],clearable=True),
        html.Div(id="flow-rate-graph-output"),
        html.Br(),
        html.Br()
            ])
    else:
        return [html.P(html.B('Select any row to display Models and Graphs'))]


##########################################################################################################################################################################################################
##########################################################################################################################################################################################################


##########################################################################################################################################################################################################
##########################################################################################################################################################################################################


# Pump Calculator Functions

output_df = []

def extract_points(filename):
    """
    This function takes out points in the text files
    and converts them into two lists for x and y values.

    Parameters:
    filename : Location of the text file

    Returns:
    Tuple of two lists containing x,y values. Also, if the file does not exist
    or the points are not in floating point form, the function returns empty values.
    """
    rnge = []
    try:
        filename = "\\\\".join(filename.split("\\"))
    except:
        print(filename,"throws error")
    try:
        if not (".txt" in filename):
            return [], [], rnge
    except:
        pass
    x, y = [], []
    try:
        with open(filename) as f:
            for line in f.readlines():
                x.append(float(line.split("\t")[0].strip(" \n")))
                y.append(float(line.split("\t")[1].strip(" \n")))
        x = np.array(x)
        y = np.array(y)
        rnge = [min(y),max(y)]
        return x, y, rnge 
    except:
        print(filename, "File does not exist in the location")
        return [], [], rnge


@app.callback([Output('output-div', 'children'),
     Output("graph-dropdown", "children")],[Input('submit-button','n_clicks')],[State({"type":"user-input","index":dash.ALL},"value")])
def update_output(n_clicks, values):

    """
    Callback function to handle the input outputs of the 
    pump calculator through data handling from the
    dataframe loaded above.

    The function responds when the user clicks on the submit button and has
    values for both the speed and pressure.

    Parameters:
    n_clicks: Number of clicks on the submit button. Required to initialize
    value1: Value entered in the pumping speed input box
    value2: Value entered in the pressure input box.

    Returns:
    Outputs to the placeholder elements for Table output and Graph dropdown.
    AgGrid for the table output and dcc.dropdown for the Graph dropdown
    with the models from the available models for the given configuration.
    """
    global default_graph_models, find_a_pump
    default_graph_models = []
    column1 = 'Pumping speed m3/hr '

    column2 = 'Ult pressure(mTorr) '
    if n_clicks and all(values):
        if len(values)==2:
            value1 = float(values[0])
            value2 = float(values[1])
        elif len(values)==1:
            model = values[0]
            # zeroed_df = pump_calc_df[(pump_calc_df[["Total Equivalent Energy", "N2 slm", "PCW typ lpm"]]==0).any(axis=1)]
            row = pump_calc_df[pump_calc_df["Model Name "] == model].iloc[0]
            value1 = row['Pumping speed m3/hr ']
            value2 = row['Ult pressure(mTorr) ']

        pump_calc_df[column1].astype(float)
        pump_calc_df[column2].astype(float)

        filtered_df = pump_calc_df[(((pump_calc_df[column1] <= (value1 * 1.2)) &
                            (pump_calc_df[column1] >= (value1 * 0.8))) &
                            ((pump_calc_df[column2] <= (value2 * 1.5)) &
                            (pump_calc_df[column2] >= (value2 * 0.5))))] # Selection range: +-20% range for Pumping speed and +-50% range for Pressure.
        filtered_df = filtered_df.sort_values('Ult pressure(mTorr) ')
        if filtered_df.empty:
            return [[
                html.Br(),
                html.Div(id='desc-div',
                        children="No possible pumps from the database",
                        style={
                            'textAlign': 'center',
                        })
            ], []] # Default output for no possible match.

        else:

            filtered_df[[
                "N2 kWh/year", "DE KWh/ year ", "PCW KWh/year ", "Total Equivalent Energy"
            ]] = filtered_df[[
                "N2 kWh/year", "DE KWh/ year ", "PCW KWh/year ", "Total Equivalent Energy"
            ]].astype(int) 
            # Converting the columns to integer type and rounding to 2 decimal places.
            filtered_df.round(2)
            filtered_df[["Total Equivalent Energy"]].round(0)
        #   print("COLUMNS",filtered_df.columns)
            dropdown_options = [{
                "label": model,
                "value": model
            } for model in set(filtered_df['Model Name '])]

            filtered_df["S.No"] = 1
            # print("COLUMNS AGAIN \n",filtered_df.columns)
            filtered_df = filtered_df[["Supplier ","Model Name ","Pumping speed m3/hr ","Ult pressure(mTorr) ","Total Equivalent Energy","Inlet ","exhaust ","PCWmin lpm","Power at ultimate KW ","Heat Balance","N2 kWh/year","DE KWh/ year ","PCW KWh/year "]]
            filtered_df.fillna(0)
            zero_rows = filtered_df[filtered_df["Total Equivalent Energy"]==0]
            filtered_df.drop(filtered_df[filtered_df["Total Equivalent Energy"]==0].index,axis=0,inplace=True)
            filtered_df = filtered_df.sort_values('Total Equivalent Energy') # Ordering based on the Total Equivalent Energyin the increasing order.
            filtered_df = pd.concat([filtered_df,zero_rows],axis=0)
            filtered_df[filtered_df==0] = "N/A"
            filtered_df[list(filtered_df.columns[:])] = filtered_df[list(filtered_df.columns[:])].astype(str)
            filtered_df = filtered_df[["Supplier ","Model Name ","Pumping speed m3/hr ","Ult pressure(mTorr) ","Total Equivalent Energy","Inlet ","exhaust ","PCWmin lpm","Power at ultimate KW ","Heat Balance","N2 kWh/year","DE KWh/ year ","PCW KWh/year "]]
            # print(filtered_df[["Max power kVA"]])
            # Style Conditions to get the first row green
            styleConditions = {
                            'styleConditions': [
                            {'condition': "params.rowIndex === 0",
                                                        'style': {
                                                            'backgroundColor': 'green',
                                                            'color': 'white'
                                                        }}
                            ]}

            if find_a_pump:
                model = values[0]
                row = filtered_df[filtered_df["Model Name "] == model].iloc[0]
                filtered_df.drop(filtered_df[filtered_df["Model Name "] == model].index[0],axis=0,inplace=True)
                v = dict(row)
                top_df = pd.DataFrame({k:[v[k]] for k in v})
                filtered_df = pd.concat([top_df,filtered_df],axis=0)
                # But, in case if we have a model for that, we replace the first row's color with gray and colour the second one green instead.
                styleConditions = {
                                'styleConditions': [{
                                    'condition': "params.rowIndex === 1",
                                    'style': {
                                        'backgroundColor': 'green',
                                        'color': 'white'
                                    }
                                },
                                {'condition': "params.rowIndex === 0",
                                                            'style': {
                                                                'backgroundColor': 'gray',
                                                                'color': 'white'
                                                            }}
                                ,
                                ]}
                default_graph_models.extend(list(filtered_df["Model Name "].iloc[:2].values))
            if not default_graph_models:
                default_graph_models.append(filtered_df["Model Name "].iloc[:2].values[0])
            return [[
            # AgGrid for the table with styling for the first row as green.
                dag.AgGrid(id='table-div',defaultColDef={ "filter": True},
                            rowData=filtered_df.to_dict('records'),
                            columnDefs=[{
                                "headerName": "S.No",
                                "width": "100px",
                                "valueGetter": {
                                    "function": "params.node.id"
                                },
                            }] +[{
                                "field": column,
                                "width": f"{len(column)*13.5}px"
                            } for column in filtered_df.columns[1:] if column!='N2 slm']+ [{
                            "width" : "130px","field":"N2 slm"}],
                            dashGridOptions={"animateRows": False},
                            getRowStyle=styleConditions
                            ),
                html.Br(),
                html.Div(id='desc-div',
                        children="""The models are aligned in Ascending Order with respect to green star rating.
                                    Highlighted green row represents the most sustainable model. Gray row is the selected model highlighted for comparision.""",
                        style={
                            'textAlign': 'center',
                            'color': 'black',
                            'fontSize': 15
                        })
            ],
                    [html.H3("Pump Curves"),
                                dcc.Dropdown(id='graph-select-dropdown',
                                                options=dropdown_options,
                                                multi=True,
                                                value=[],
                                                clearable=False,
                                                style={"width":"250px"})]]
    else:
        return [[], []]


@app.callback(Output('graph-div', 'children'),
              Input('graph-select-dropdown', 'value'))
def update_selected_values(selected_values):
    """
    This function plots the graphs for the models
    selected through the dropdown.

    Callback function that responds to the selection
    from the dropdown

    Parameters:
    selected_values: List of selected values from the dropdown

    Returns:
    Graph output from the selected models.
    """ 
    global default_graph_models
    if len(selected_values) + len(default_graph_models):

        # Setting up a dataframe from the excel sheet 
        # that contains the model name corresponding to
        # the location of the text file for the data points.
        file_location = r"..\..\..\..\Downloads\pumpModels_plotsdata.xlsx"
        wb = openpyxl.load_workbook(file_location)
        sheet = wb.worksheets[0] #[1]
        data = sheet.values
        cols = next(data)[0:]
        data = list(data)
        data = (islice(r, 0, None) for r in data)
        # data_df = pd.DataFrame(data, columns=cols)
        data_df = pd.read_excel(file_location) 
        selected_values = selected_values + default_graph_models
        
        indices = []

        for val in selected_values:
            try:
                indices.append((np.where(data_df['Model Name '] == val))[0][0])
            except:
                pass

        ranges = []
        fig = make_subplots(specs=[[{"secondary_y": False}]]) # Canvas for plots
        for model in data_df.loc[indices]['file location ']:
            data = extract_points(model)
            # print(data)
            if not (ranges and data[2]):
                ranges.extend(data[2])
                # print("HI")
            else:
                ranges[0] = min(ranges[0],data[2][0])
                ranges[1] = max(ranges[1],data[2][1])
            fig.add_trace(go.Scatter(x=data[0], y=data[1], name=selected_values[0]))
            selected_values.pop(0)
            # print(ranges)
        fig['layout'].update(height=600,
                            width=800,
                            title='Performance Curve',
                            )
        # (ticklen=6, tickcolor="black", tickmode='auto', nticks=10, showgrid=True)
        fig.update_xaxes(title_text="Inlet pressure/Torr",type='log',range=[-3,3])
        fig.update_yaxes(title_text="Pumping speed/m3/hr",range=ranges)
        return dcc.Graph(figure=fig)  
    # elif default_graph_models:
    else:
        return "No values selected"


#####################################################################################################
#####################################################################################################

app.layout = html.Div([
    html.Div(children=[
        html.Div(
            children=[
            description_card(), generate_control_card(),
            ], style={"align-items":"center","margin":"10px"}
        ),
            ]),
])
if __name__ == '__main__':
    app.run(host='127.0.0.1',port='8070',debug=True)
