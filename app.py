import streamlit as st
import pandas as pd
import plotly.io as pio
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import gender_guesser.detector as gender
import itertools
from scipy.stats import pearsonr, pointbiserialr, chi2_contingency, spearmanr, f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.write("### Предварительный просмотр данных")
    st.dataframe(df.head())
    return df

def prepare_data_for_tests(df):
    df_ = df.copy()
    df_['last_review_year'] = pd.to_datetime(df_['last_review']).dt.year
    columns_to_remove = ['id', 'name', 'host_name', 'host_id', 'neighbourhood', 'last_review']
    df_ = df_.drop(columns=columns_to_remove, errors='ignore')
    categorical_cols = ['neighbourhood_group', 'room_type', 'host_name_gender', 'last_review_year']
    df_ = pd.get_dummies(df_, columns=categorical_cols, drop_first=False)
    return df_

def handle_missing_values(df):
    st.sidebar.write("### Обработка пропущенных значений")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            option = st.sidebar.radio(
                f"Как обработать пропущенные значения в '{col}'?", 
                ('Удалить строки', 'Заполнить средним', 'Заполнить медианой', 'Заполнить 0', 'Заполнить произвольным значением'), 
                index=0
            )
            if option == 'Удалить строки':
                df = df.dropna(subset=[col])
            elif option == 'Заполнить средним':
                df[col] = df[col].fillna(df[col].mean())
            elif option == 'Заполнить медианой':
                df[col] = df[col].fillna(df[col].median())
            elif option == 'Заполнить 0':
                df[col] = df[col].fillna(0)
            elif option == 'Заполнить произвольным значением':
                custom_value = st.sidebar.number_input(f"Произвольное значение для '{col}'", value=0)
                df[col] = df[col].fillna(custom_value)
    return df

def detect_and_remove_outliers(df):
    st.sidebar.write("### Выявление и удаление выбросов")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if st.sidebar.checkbox(f"Выявить выбросы в '{col}'"):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            st.sidebar.write(f"Выбросы удалены из '{col}'")
    return df

def detect_gender(df):
    d = gender.Detector()
    df['host_name_gender'] = df.host_name.apply(lambda x: d.get_gender(str(x).capitalize()))
    df['host_name_gender'] = df['host_name_gender'].replace({
        'mostly_female': 'female', 
        'mostly_male': 'male', 
        'andy': 'unknown'
    })
    return df

def show_visualizations(df):
    st.write("### Визуализации")

    if st.checkbox("Показать логарифмическую гистограмму"):
        st.write("### Логарифмическое распределение свойств хостов")
        hist = px.histogram(df.groupby('host_id')['host_id'].count(), 
                            title='Логарифмическое распределение свойств хостов', 
                            log_y=True, log_x=True)
        st.plotly_chart(hist)

    if st.checkbox("Показать диаграмму рассеяния"):
        st.write("### Диаграмма рассеяния отзывов в месяц против цены")
        scatter = px.scatter(df, 'reviews_per_month', 'price', 
                             color='neighbourhood_group', 
                             log_y=True, opacity=0.6)
        st.plotly_chart(scatter)

    if st.checkbox("Показать временной ряд"):
        st.write("### Временной ряд отзывов с течением времени")
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
        df_clean = df.dropna(subset=['last_review', 'reviews_per_month'])
        df_clean['reviews_per_month'] = pd.to_numeric(df_clean['reviews_per_month'], errors='coerce')
        time_series = df_clean.groupby(df_clean['last_review'].dt.to_period('M'))['reviews_per_month'].sum()
        time_series = time_series.reset_index()
        time_series['last_review'] = time_series['last_review'].dt.to_timestamp()
        fig = px.line(time_series, x='last_review', y='reviews_per_month', title='Тренд месячных отзывов')
        st.plotly_chart(fig)

    if st.checkbox("Показать ящичную диаграмму цен"):
        st.write("### Распределение цен по типу комнаты")
        box_plot = px.box(df, x='room_type', y='price', color='neighbourhood_group')
        st.plotly_chart(box_plot)

    if st.checkbox("Показать карту объявлений"):
        st.write("### Объявления Airbnb в Нью-Йорке")
        map_fig = px.scatter_mapbox(
            df, lat="latitude", lon="longitude", color="neighbourhood", size="price", 
            hover_name="name", zoom=11, height=700, title="Объявления Airbnb по районам"
        )
        map_fig.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(map_fig)

    if st.checkbox("Показать тепловую карту корреляций"):
        df_ = prepare_data_for_tests(df)
        st.write("### Тепловая карта корреляций")
        plt.figure(figsize=(45, 40))
        sns.heatmap(df_.corr(numeric_only=True), annot=True, cmap="YlGnBu")
        st.pyplot(plt)

        st.write("#### Только значимые взаимосвязи для каждого признака")
        for i in range(30):
            try:
                plt.subplots(figsize=(30, 1))
                col = df_.columns[i]
                df__ = df_.corr(numeric_only=True).round(1).iloc[[i]].T.sort_values(col)
                df__ = df__[abs(df__)>=0.3].dropna().T

                dataplot = sns.heatmap(df__, cmap="YlGnBu", annot=True)
                st.pyplot(plt)
            except:
                continue

def run_statistical_test(df, alpha=0.05):
    st.write("### Статистические тесты")
    columns = df.columns
    selected_columns = st.multiselect("Выберите столбцы для статистических тестов", columns, default=columns[:2])

    if len(selected_columns) >= 2:
        pairs = itertools.combinations(selected_columns, 2)
        results = {}

        for col1, col2 in pairs:
            col1_type = df[col1].dtype
            col2_type = df[col2].dtype
            col1_is_binary = df[col1].nunique() == 2
            col2_is_binary = df[col2].nunique() == 2

            try:
                if col1_type in ['int64', 'float64'] and col2_type in ['int64', 'float64']:
                    corr, p_value = pearsonr(df[col1], df[col2])
                    test_name = 'Корреляция Пирсона'

                elif (col1_is_binary and col2_type in ['int64', 'float64']) or (col2_is_binary and col1_type in ['int64', 'float64']):
                    if col1_is_binary:
                        corr, p_value = pointbiserialr(df[col1], df[col2])
                    else:
                        corr, p_value = pointbiserialr(df[col2], df[col1])
                    test_name = 'Точечно-бисериальная корреляция'

                elif col1_is_binary or col2_is_binary:
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    corr = chi2 
                    test_name = 'Тест хи-квадрат'

                elif col1_type in ['int64', 'float64'] and col2_type in ['int64', 'float64']:
                    corr, p_value = spearmanr(df[col1], df[col2])
                    test_name = 'Корреляция Спирмена'

                else:
                    continue
               
                results[(col1, col2)] = {'тест': test_name, 'корреляция': corr, 'p-значение': p_value}
                # scatter = px.scatter(pd.DataFrame(results), 'reviews_per_month', 'price', 
                #                      color='neighbourhood_group', 
                #                      log_y=True, opacity=0.6)
            except Exception as e:
                st.warning(f"Ошибка при тестировании {col1} и {col2}: {e}")

        df_ = pd.DataFrame(results).T
        for i in df_['тест'].unique():
            st.write(px.scatter(df_[df_['тест']==i],x='корреляция',y='p-значение',title=f'Тест: {i}'))


       
        if results:
            results_df = pd.DataFrame.from_dict(results, orient='index')
            if 'p_value' in results_df.columns:
                results_df = results_df.sort_values(by='p_value')
            st.dataframe(results_df)
            return results_df.reset_index()
        else:
            st.write("Значимых результатов не найдено.")

    else:
        st.write("Выберите хотя бы два столбца для тестирования.")

def show_correlation_heatmap(df):
    st.write("### Тепловая карта корреляций")
    plt.figure(figsize=(45, 40))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu")
    st.pyplot(plt)

def train_predictive_model(df):
    st.write("### Прогнозирование цены на основе параметров")
    predictors = st.multiselect("Выберите предикторы для модели", df.columns.tolist())
    target = st.selectbox("Выберите целевую переменную", df.columns.tolist())

    if predictors and target:
        X = df[predictors]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"Коэффициент детерминации (R^2): {r2}")
        st.write(f"Среднеквадратичная ошибка (MSE): {mse}")

       
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel('Фактические значения')
        ax.set_ylabel('Прогнозируемые значения')
        ax.set_title('Фактические vs Прогнозируемые значения')
        st.pyplot(fig)


def main():
    st.title('Анализ объявлений Airbnb')

    uploaded_file = st.sidebar.file_uploader("Загрузите данные в формате CSV", type="csv")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = handle_missing_values(df)
        df = detect_and_remove_outliers(df)
        df = detect_gender(df)

        show_visualizations(df)

        df_for_tests = prepare_data_for_tests(df)

        run_statistical_test(df_for_tests)

        train_predictive_model(df_for_tests)

if __name__ == "__main__":
    main()
