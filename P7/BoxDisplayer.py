
import streamlit as st
import numpy as np
import datetime

INT_MAX_VALUE = 9007199254740991

class BoxDisplayer:
	def __init__(self, unique_items_dict, streamlit_pos, **kwargs):
		self.__dict__.update(kwargs)
		self.__unique_items_dict = unique_items_dict
		self.__streamlit_pos = streamlit_pos


	def __display_string_box(self, feature, title, _value):
		if isinstance(_value, str):
			index = self.__unique_items_dict[feature].index(_value)
		else:
			index = 0
		value = st.selectbox(title,
							 self.__unique_items_dict[feature],
							 index=index)
		if value == 'undefined':
			value = np.nan
		return value


	def __display_date_box(self, title, _value):
		if not np.isnan(_value):
			date = datetime.date.today() + datetime.timedelta(days=int(_value))
			value = (st.date_input(title, date) - datetime.date.today()).days
		else:
			date = datetime.date(1900, 1, 1)
			value = (st.date_input(title, date) - datetime.date.today()).days
			if value == (datetime.date(1900, 1, 1) - datetime.date.today()).days:
				value = np.nan
		return value


	def __display_bool_box(self, title, _value):
		bool_list = [0, 1, 'undefined']
		if not np.isnan(_value):
			index = int(_value)
			value = st.selectbox(title,
								 [0, 1],
								 index=index)
		else:
			index = 2
			value = st.selectbox(title,
								 bool_list,
								 index=index)
			value = value if value != 'undefined' else np.nan
		return value


	def __get_pow(self, val):
		val_sc = np.format_float_scientific(val)
		mantissa, exp = val_sc.split('e')
		return int(exp)


	def __display_float_box(self, title, _value):
		if not np.isnan(_value):
			value = np.float64(_value)
			value_power = self.__get_pow(value)
			step = np.float64(pow(10, value_power-1))
			if step < 1:
				format_ = "%."+str(-value_power+1)+"f"
			else:
				format_ = "%.0f"
			value = st.number_input(title,
									value=value,
									step=step,
									format=format_)
		else:
			value = st.number_input('FLOAT',
									value=sys.float_info.max,
									step=np.float64(0.1),
									format="%.1f")
			value = value if value != sys.float_info.max else np.nan
		return value


	def __display_int_box(self, title, _value):
		if not np.isnan(_value):
			value = st.number_input(title,
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


	def __display_box(self, feature, title, value, type_):
		# Case column_type is STRING
		if type_ == 'STRING':
			value = self.__display_string_box(feature, title, value)
		# Case column_type is DATE
		elif type_ == 'DATE':
			value = self.__display_date_box(title, value)
		# Case column_type is BOOL
		elif type_ == 'BOOL':
			value = self.__display_bool_box(title, value)
		# Case column_type is FLOAT
		elif type_ == 'FLOAT':
			value = self.__display_float_box(title, value)
		# Case column_type is INT
		elif type_ == 'INT':
			value = self.__display_int_box(title, value)
		# UNEXPECTED CASE
		else:
			print('ERROR: TYPE ', column_type, 'NOT TAKE IN CHARGE')
		return value


	def display_editable_feature(
			self, key_df, df, feature, description_text, type_):

		# Check df is not empty
		if len(df) != 0:
			# Get value in df for considered feature
			value = df.loc[0, feature]
		else:
			value = np.nan

		with self.__streamlit_pos:
			df.loc[0, feature] = self.__display_box(
				feature, description_text, value, type_)
			return

	def display_fixed_feature(self, key_df, df, feature, description, type_):

		# Check df is not empty
		if len(df) != 0:
			# Get value in df for considered feature
			value = df.loc[0, feature]
		else:
			value = np.nan

		# If value is not defined
		with self.__streamlit_pos:
			if isinstance(value, float) and np.isnan(value):
				st.metric(description, "Undefined")
			# Case column_type is STRING
			elif type_ == 'STRING':
				st.metric(description, "{:s}".format(value))
			# Case column_type is DATE
			elif type_ == 'DATE':
				# Convert value to date
				date = datetime.date.today() + datetime.timedelta(days=int(value))
				st.metric(description, "{:n}".format(date))
			# Case column_type is FLOAT
			elif type_ == 'FLOAT':
				value = np.float64(value)
				st.metric(description, "{:g}".format(value))
			# Case column_type is INT or BOOL
			elif type_ == 'INT' or type_ == 'BOOL':
				st.metric(description, "{:n}".format(value))
			# UNEXPECTED CASE
			else:
				print('ERROR: TYPE ', type_, 'NOT TAKE IN CHARGE')

		return
