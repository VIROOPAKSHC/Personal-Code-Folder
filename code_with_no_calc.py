import dash
from dash import html, dcc, Input, Output, State
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
import numpy as np

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.UNITED, dbc.icons.BOOTSTRAP],suppress_callback_exceptions=True)
k_dict = {0:0.017,1:0.05,2:0.083,4:0.073,6:0.056,8:0.042,10:0.034,12:0.029,14:0.026}
df = pd.read_excel("PumpModels4.1.xlsx")

def description_card():
  return html.Div(
      id="description-card",
      children=[
          html.H5("Pump Models",style={"font-weight":"bold"}),
          html.H3("Welcome to the Pump Finder Dashboard",style={"font-weight":"bold"}),
          html.Div(
              id="intro",
              children="Explore pump models, based on the range of pumping speed and pressure. Click on the model names from the dropdown to visualize pump curves.",
          ),
          html.Br()
      ]
  )


app.layout = html.Div([
    description_card(),
    html.B("Select - Type of your flow"),
    dcc.Tabs(id='output-tabs', value='tab-1', children=[
        dcc.Tab(label='Viscous', value='tab-1',style={"fontWeight":"bold"},selected_style={"fontWeight":"bold"}),
        dcc.Tab(label='Molecular', value='tab-2',style={"fontWeight":"bold"},selected_style={"fontWeight":"bold"}),
    ]),
    html.Div(id='output-container',style={"padding":"10px"}),
    html.Div(id='output-container-2',style={"padding":"10px"})
],style={"padding":"10px"})


@app.callback(
    Output('output-container', 'children'),
    Input('output-tabs', 'value')
)
def update_output(selected):
    if selected == "tab-1":
        return ([
            html.Div([dcc.Markdown("** Enter the target Pressure **",style={"width":"200px"}),
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
            html.Button("Submit",id='submit-button',n_clicks=0),
        ])
    else:
        return html.Div([
            html.Br(),
            html.B('Molecular Flow was clicked!'),
            html.Br()
        ])

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

# @app.callback(Output('pumping-speed-2','value'),Input('pumping-speed-1','value'))
# def set_value(pump):
#     return pump

@app.callback(
    Output('output-container-2', 'children', allow_duplicate=True),
    Input('submit-button', 'n_clicks'),
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
        
        filtered_df  = df
        # try:
        #     filtered_df = df[(df["Pumping speed m3/hr "] >= pump_speed*0.9) & (df["Pumping speed m3/hr "] <= (pump_speed*1.1))]
        # except:
        #     pass
        filtered_df.round(2)
        cols = df.columns
        
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
        outputs.append(html.Br())
        target_S_eff = (volume/tp)*math.log(760/pressure)
        target_S_eff = round(target_S_eff,4)
        # outputs.append(html.P(f"Effective Speed calculation = ({volume}/{tp}) * log10(({760})/{pressure}) = {target_S_eff}"))
        # outputs.append(html.B(f"Effective Speed S_eff = {target_S_eff} L/S"))
        outputs.append(html.Br())
        outputs.append(html.B(f"Table :"))
        outputs.append(html.Br())
        target_S = round((target_S_eff*C)/(C-target_S_eff),4)*3.6
        # outputs.append(html.P(f"Pumping Speed for Comparision = ({target_S_eff}*{C})/({C}-{target_S_eff}) = {target_S} m3/hr"))
        slider = slider/100 #put a slider
        filtered_df = filtered_df[(filtered_df["Pumping speed m3/hr "] >= (1-slider)*target_S) & (filtered_df["Pumping speed m3/hr "] <= (1+slider)*target_S)]
        filtered_df["Pump_DownTimes"] = [calculate_downtimes_pipes(model,pressure,volume) for model in filtered_df["Model Name "]]
        filtered_df = filtered_df.sort_values("Total Equivalent Energy")
        # cols = filtered_df.columns[0:2].to_list() + ["Pump_DownTimes"] + filtered_df.columns[2:-1].to_list()
        cols = ["Supplier ","Model Name ","Pumping speed m3/hr ","Ult pressure(mTorr) ","Pump_DownTimes","Total Equivalent Energy","Inlet ","exhaust ","PCWmin lpm","Power at ultimate KW ","Heat Balance","N2 kWh/year","DE KWh/ year ","PCW KWh/year "]
        columns = [{"field":col} for col in cols]
        filtered_df = filtered_df[cols]
        data = filtered_df.to_dict('records')
        grid = dag.AgGrid(
            id='table-div',
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
        outputs.append(html.Div(id="table-div", children=[grid],style={"padding":"10px"}))
        outputs.append(html.Div(id="Pumping-Speed-Output"))
        return outputs

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
        selected_row = df[df["Pumping speed m3/hr "] == selected_rows[0]["Pumping speed m3/hr "]].sort_values("Total Equivalent Energy")
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

        selected_row = df[df["Pumping speed m3/hr "] == selected_rows[0]["Pumping speed m3/hr "]].sort_values("Total Equivalent Energy")
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


# Run the app
if __name__ == '__main__':
    app.run(host='127.0.0.1',port='8050',debug=True)
