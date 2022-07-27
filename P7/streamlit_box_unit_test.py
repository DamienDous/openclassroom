import pandas as pd
import streamlit as st
import numpy as np
import datetime
import sys

INT_MAX_VALUE = 9007199254740991
# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

def get_pow(val):
	val_sc = np.format_float_scientific(val)
	mantissa, exp = val_sc.split('e')
	return int(exp)

def display_string_box(_value):
	if isinstance(_value, str):
		index=int(float(_value))
	else:
		index=0
	value = st.selectbox('STRING',
						 ['undefined','1', '2'],
						 index=index)
	if value == 'undefined':
		value = np.nan
	return value

def display_date_box(_value):
	if not np.isnan(_value):
		date = datetime.date.today() + datetime.timedelta(days=int(_value))
		value = (st.date_input('DATE', date) - datetime.date.today()).days
	else:
		date = datetime.date(1900, 1, 1)
		value = (st.date_input('DATE', date) - datetime.date.today()).days				
		if value == (datetime.date(1900, 1, 1) - datetime.date.today()).days:
			value = np.nan				
	return value


def display_bool_box(_value):
	bool_list = [0, 1, 'undefined']
	if not np.isnan(_value):
		index = _value
		value = st.selectbox('BOOL',
							 [0, 1],
							 index=index)
	else:
		index = 2
		value = st.selectbox('BOOL',
							 bool_list,
							 index=index)
		value = value if value != 'undefined' else np.nan
	return value


def display_float_box(_value):
	if not np.isnan(_value):
		value_power = get_pow(_value)
		step = np.float64(pow(10, value_power-1))
		if step < 1:
			format_ = "%."+str(-value_power+1)+"f"
		else:
			format_ = "%.0f"
		value = st.number_input('FLOAT',
								value=_value,
								step=step,
								format=format_)
	else:
		value = st.number_input('FLOAT',
								value=sys.float_info.max,
								step=np.float64(0.1),
								format="%.1f")
		value = value if value != sys.float_info.max else np.nan
	return value

def display_int_box(_value):	
	if not np.isnan(_value):
		value = st.number_input('INT',
								value=np.int64(_value),
								step=np.int64(1),
								format="%d")
	else:
		value = st.number_input('INT',
								value=INT_MAX_VALUE,
								step=np.int64(1),
								format="%d")
		value = value if value != INT_MAX_VALUE else np.nan
	return value
	

def display_box(data):
	for index in data.index:
		# Case column_type is STRING
		if data.loc[index, 'Type'] == 'STRING':
			data.loc[index, 'Value'] = display_string_box(data.loc[index, 'Value'])
		# Case column_type is DATE
		elif data.loc[index, 'Type']  == 'DATE':
			data.loc[index, 'Value'] = display_date_box(data.loc[index, 'Value'])
		# Case column_type is BOOL
		elif data.loc[index, 'Type']  == 'BOOL':
			data.loc[index, 'Value'] = display_bool_box(data.loc[index, 'Value'])
		# Case column_type is FLOAT
		elif data.loc[index, 'Type'] == 'FLOAT':
			data.loc[index, 'Value'] = display_float_box(data.loc[index, 'Value'])
		# Case column_type is INT
		elif data.loc[index, 'Type'] == 'INT':
			data.loc[index, 'Value'] = display_int_box(data.loc[index, 'Value'])
		# UNEXPECTED CASE
		else:
			print('ERROR: TYPE ', column_type, 'NOT TAKE IN CHARGE')
	return data

data_not_good = pd.DataFrame(
	 [['STRING', np.nan], 
	  ['DATE', np.nan], 
	  ['BOOL', np.nan], 
	  ['FLOAT', np.nan], 
	  ['INT', np.nan]
	 ], 
	columns=['Type', 'Value'])

data_good = pd.DataFrame(
	 [['STRING', '2'], 
	  ['DATE', -10], 
	  ['BOOL', 0], 
	  ['FLOAT', 10033.0011], 
	  ['INT', 3]
	 ], 
	columns=['Type', 'Value'])

display_box(data_good)
display_box(data_not_good)

select_btn = st.button('Select')

if select_btn:
	st.write(data_good.astype(str))
	st.write(data_not_good.astype(str))
