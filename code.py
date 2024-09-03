import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_ag_grid as dag
import os

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
    html.Div([
        html.Button('Viscous Flow', id='button-1', n_clicks=0),
        html.Button('Molecular Flow', id='button-2', n_clicks=0, style={'margin-left': '25px'}),
    ],style={"padding":"10px"}),
    html.Div(id='output-container',style={"padding":"10px"}),
    html.Div(id='output-container-2',style={"padding":"10px"})
],style={"padding":"10px"})


@app.callback(
    Output('output-container', 'children'),
    Input('button-1', 'n_clicks'),
    Input('button-2', 'n_clicks')
)
def update_output(btn1_clicks, btn2_clicks):
    if btn1_clicks > 0:
        return html.Div([
            html.B("Enter the Pump Down time (in seconds)"),
            html.Br(),
            dcc.Input(id='input-1', type='number', placeholder='Pump Down Time'),
            html.Br(),
            html.B("Enter the target Pressure (in mTorr)"),
            html.Br(),
            dcc.Input(id='input-2', type='number', placeholder='Target Pressure'),
            html.Br(),
            html.B("Enter the Volume of the Chamber (in L)"),
            html.Br(),
            dcc.Input(id='input-3', type='number', placeholder='Chamber Volume'),
            html.Br(),
            html.B("Enter the Pipe dimensions"),
            html.Br(),
            dash_table.DataTable(
                id='input-table',
                columns=[
                    {'name': 'Length (in cm)', 'id': 'length', 'type': 'numeric'},
                    {'name': 'Diameter (in cm)', 'id': 'diameter', 'type': 'numeric'},
                ],
                data=[{'length': '', 'diameter': ''}],
                editable=True,
                row_deletable=True,
                style_cell={'width': '70%', 'textAlign': 'center'}
            ),
            html.Br(),
            html.Button('Add Row', id='add-row-button', n_clicks=0),
            html.Br(),
            html.B("Enter the Bend dimensions"),
            html.Br(),
            dash_table.DataTable(
                id='input-table-2',
                columns=[
                    {'name': 'Radius (in cm)', 'id': 'radius', 'type': 'numeric'},
                    {'name': 'Diameter (in cm)', 'id': 'diameter', 'type': 'numeric'},
                ],
                data=[{'radius': '', 'diameter': ''}],
                editable=True,
                row_deletable=True,
                style_cell={'width': '70%', 'textAlign': 'center'}
            ),
            html.Br(),
            html.Button('Add Row', id='add-row-button-2', n_clicks=0),
            html.Br(),
            html.B("Enter or Select Pumping Speed"),
            html.Br(),
            html.Div([dcc.Dropdown(id='pumping-speed-1',options=df['Pumping speed m3/hr '].unique(),placeholder='Select Speed',style={"width":"250px"}),html.Br(),
                      dcc.Input(id='pumping-speed-2',type='number',placeholder='Enter Pumping Speed',style={"width":"250px"})]),
            html.Br(),
            html.B("Enter the Flow Rate in Torr-L/s"),
            html.Br(),
            dcc.Input(id="flow-rate",type="number",placeholder="Flow Rate"),
            html.Br(),
            html.Button("Submit",id='submit-button',n_clicks=0),
            html.Br()
        ])
    elif btn2_clicks > 0:
        return html.Div([
            html.Br(),
            html.B('Button 2 was clicked!'),
            html.Br()
        ])
    else:
        return html.Div()
    
@app.callback(
    Output('input-table', 'data'),
    Input('add-row-button', 'n_clicks'),
    State('input-table', 'data'),
    State('input-table', 'columns')
)
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows

@app.callback(
    Output('input-table-2', 'data'),
    Input('add-row-button-2', 'n_clicks'),
    State('input-table-2', 'data'),
    State('input-table-2', 'columns')
)
def add_row_2(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows

@app.callback(Output('pumping-speed-2','value'),Input('pumping-speed-1','value'))
def set_value(pump):
    return pump

@app.callback(
    Output('output-container-2', 'children', allow_duplicate=True),
    Input('submit-button', 'n_clicks'),
    Input('input-1', 'value'),
    Input('input-2', 'value'),
    Input('input-3', 'value'),
    State('pumping-speed-2','value'),
    State('input-table', 'data'),
    State('input-table-2','data'),
    State('flow-rate','value'),
    prevent_initial_call=True
)
def calculate_downtimes(n_clicks, input1, input2, input3, pump_speed, table_data,table_data_2,flow_rate):
    if n_clicks > 0:
        tp = input1
        pressure = input2
        volume = input3
        C = 0
        mu = 0.0179
        p_ = round((pressure+760)/2,3)
        global table_values
        global table_values_2
        table_values = []
        table_values_2 = []
        PI = 3.14
        target_S_eff = (volume/tp)*math.log(760/pressure)
        
        for row in table_data:
            table_values.append([row['length'],row['diameter']])
        
        for row in table_data_2:
            table_values_2.append([row['radius'],row['diameter']])

        outputs=[]
        for l,d in table_values:
            l,d = round(l,4),round(d,4)
            c = 0.0327*(d**4/(mu*l))*p_
            c = round(c,4)
            C += 1/(c)
        
        constant_factor = (PI*1000*p_) / (128*mu*0.1)
        for r,d in table_values_2:
            K = k_dict[int(r/d)]
            r,d = round(r,4),round(d,4)
            c = constant_factor*K*((d*0.01)**3)
            c = round(c,4)
            C += 1/(c)

        C = 2/C
        outputs.append(html.B(f"Conductance Calculated : {C}"))
        C = round(C,4)
        filtered_df  = df
        try:
            filtered_df = df[(df["Pumping speed m3/hr "] >= pump_speed*0.9) & (df["Pumping speed m3/hr "] <= (pump_speed*1.1))]
        except:
            pass

        cols = df.columns
        
        if flow_rate:
            Q = []
            steady_state_df = filtered_df[filtered_df["Ult pressure(mTorr) "] <= pressure]
            for So in steady_state_df[cols[2]].unique():
                So = float(So)
                S_eff = (So*C)/(So+C)
                Q.append([So,round(S_eff*pressure,4)])
            Q.sort(key=lambda x:abs(flow_rate-x[1]))
            Q_table = pd.DataFrame(Q,columns=["Pumping Speed","Flow Rate"])
            outputs.append(html.Br())
            outputs.append(html.B(f"Flow Rates table :"))
            outputs.append(html.Br())
            columns = [{'headerName': i, 'field': i} for i in Q_table.columns]
            data = Q_table.to_dict('records')
            grid = dag.AgGrid(
                id='flow-rate-div',
                columnDefs=columns,
                rowData=data,
                columnSize="sizeToFit",
                defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 100},
                dashGridOptions={"pagination": True, "paginationPageSize": 8, "domLayout": "autoHeight",'rowSelection': "single"},
                style={"height": 500, "width": "100%"},
                className="ag-theme-alpine"
            )
            outputs.append(html.Div(id="flow-rate-div", children=[grid],style={"padding":"10px"}))
            
            outputs.append(html.Div(id='Flow-Rate-Output'))
        
        filtered_df = filtered_df[filtered_df["Ult pressure(mTorr)"] <= pressure]
        filtered_df["Effective Pumping Speed"] = 0.0
        for So in filtered_df[cols[2]].unique():
            So = float(So)
            S_eff = (So*C)/(So+C)
            S_eff = round(S_eff,4)
            filtered_df.loc[filtered_df["Pumping speed m3/hr "] == So,"Effective Pumping Speed"] = S_eff
        
        outputs.append(html.Br())
        outputs.append(html.B(f"Effective Speeds Table :"))
        outputs.append(html.Br())
        filtered_df = filtered_df[(filtered_df["Effective Pumping Speed"] >= 0.5*target_S_eff) & (filtered_df["Effective Pumping Speed"] <= 1.5*target_S_eff)]
        filtered_df = filtered_df.sort_values("Total Equivalent Energy")
        cols = filtered_df.columns[0:2].to_list() + ["Effective Pumping Speed"] + filtered_df.columns[2:-1].to_list()
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

def extract(filename,location,target_pressure):
    filename = os.path.join(os.path.join(os.getcwd(),location),filename)
    values = [row.strip("\n").split("\t") for row in open(filename).readlines()]
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
def display_selected_row(selected_rows,pressure,volume):
    if selected_rows:
        selected_row = df[df["Pumping speed m3/hr "] == selected_rows[0]["Pumping speed m3/hr "]].sort_values("Total Equivalent Energy")
        outputs = [
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
        ]
        model_name = selected_row["Model Name "].values[0]
        filename = f"{model_name}.txt"
        location = "GRAPH_DATA"
        print(os.listdir(),filename)
        if filename in os.listdir(location):
            pressures,speeds = extract(filename,location,pressure)
            pressures = [760] + list(pressures) + [pressure]
            speeds = [-1] + list(speeds) + [selected_rows[0]["Pumping speed m3/hr "]]
            pfs = []
            conductances = []
            mu = 0.0179
            PI = 3.14
            effective_speeds = []
            pump_down_times = []
            Q = []
            for i in range(len(pressures)):
                if i==0:
                    conductances.append(-1)
                    effective_speeds.append(-1)
                    pump_down_times.append(-1)
                    pfs.append(-1)
                    Q.append(-1)
                else:
                    So = speeds[i]
                    p_ = (pressures[i] + pressures[i-1])/2
                    C = 0
                    for l,d in table_values:
                        l,d = round(l,4),round(d,4)
                        c = 0.0327*(d**4/(mu*l))*p_
                        c = round(c,4)
                        C += 1/(c)
                    
                    constant_factor = (PI*1000*p_) / (128*mu*0.1)
                    for r,d in table_values_2:
                        K = k_dict[int(r/d)]
                        r,d = round(r,4),round(d,4)
                        c = constant_factor*K*((d*0.01)**3)
                        c = round(c,4)
                        C += 1/(c)

                    C = 2/C
                    conductances.append(C)
                    S_eff = (So*C)/(So+C)
                    effective_speeds.append(S_eff)
                    time = round(((volume/S_eff)*math.log(760/pressure)),4)
                    pump_down_times.append(time)
                    pf = (S_eff/(C+S_eff))*760
                    pfs.append(pf)
                    Q.append(pressures[i]*S_eff)
            another_table = pd.DataFrame({"Ult Pressure (Torr)":pressures,"Pumping speed m3/hr ":speeds,"Conductance":conductances,"Effective Speed":effective_speeds,"Pump Down Time":pump_down_times})
            columns = [{'headerName': i, 'field': i} for i in another_table.columns]
            data = another_table.to_dict('records')
            grid = dag.AgGrid(
                id='flow-rate-div',
                columnDefs=columns,
                rowData=data,
                columnSize="sizeToFit",
                defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 100},
                dashGridOptions={"pagination": True, "paginationPageSize": 8, "domLayout": "autoHeight",'rowSelection': "single"},
                style={"height": 500, "width": "100%"},
                className="ag-theme-alpine"
            )
            outputs.append(html.Div(id="individual-pressures-div", children=[grid],style={"padding":"10px"}))
            another_table_2 = pd.DataFrame({"Ult Pressure (Torr)":pressures,"Pumping speed m3/hr ":speeds,"Conductance":conductances,"Effective Speed":effective_speeds,"Pf":pfs,"Flow Rate":Q})
            columns = [{'headerName': i, 'field': i} for i in another_table_2.columns]
            data = another_table_2.to_dict('records')
            grid = dag.AgGrid(
                id='flow-rate-div-2',
                columnDefs=columns,
                rowData=data,
                columnSize="sizeToFit",
                defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 100},
                dashGridOptions={"pagination": True, "paginationPageSize": 8, "domLayout": "autoHeight",'rowSelection': "single"},
                style={"height": 500, "width": "100%"},
                className="ag-theme-alpine"
            )
            outputs.append(html.Div(id="individual-pressures-div-2", children=[grid],style={"padding":"10px"}))
            
        outputs.extend([html.B("Select model(s) for the Pressure vs Time graph"),
        html.Br(),
        dcc.Dropdown(id="graph-dropdown",options=selected_row["Model Name "].to_list(),multi=True,value=[],clearable=True),
        html.Div(id="graph-output"),
        html.Br(),
        html.Br()])
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
        selected_row = selected_row[selected_row["Ult pressure(mTorr)"] <= pressure]
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
