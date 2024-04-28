import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from catboost import CatBoostRegressor

pd.options.mode.copy_on_write = True


class WeatherModel:
    """Класс для создания прогноза погоды."""

    SIZE_Y = 30
    SIZE_X = 30

    HOURS_TRAIN = 43
    HOURS_TEST = 5
    HOURS_IN_DAY = 24

    CLOUD_COVER = "cloud_cover"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    WIND_DIR = "wind_dir"
    WIND_SPEED = "wind_speed"

    # Высота - ее предсказывать не надо
    ELEVATION = "elevation"

    TYPES_WEATHER = [TEMPERATURE, PRESSURE, HUMIDITY, WIND_SPEED, WIND_DIR, CLOUD_COVER]
    COLUMNS = [ELEVATION, TEMPERATURE, PRESSURE, HUMIDITY, WIND_SPEED, WIND_DIR, CLOUD_COVER]

    COLUMNS_PREDICT_BASE = [TEMPERATURE, HUMIDITY, WIND_SPEED]
    COLUMNS_PREDICT_CATBOOST = [PRESSURE, WIND_DIR, CLOUD_COVER]

    LEARNING_RATE_FILE = "learning_rate.csv"

    # Коэффициенты подобраны эмпирическим путем
    CLOUD_COVER_KOEF = 0.8
    WIND_DIR_KOEF = 0.7

    def __init__(self, type_model: str):
        """Инициализация класса.

        Args:
            type_model (str): Тип модели, которую будем использовать для обучения и предсказания.
            1) base_model - будем домножать на какой-нибудь коэффициент.
            2) catboost_model - предсказываем значения с помощью catboost.
            3) mix_model - предсказываем значения с помощью catboost и базовой модели.
        """
        self.type_model = type_model
        self.data_elevation = None

        self.params_catboost = {"learning_rate": 0.7,
                                "iterations": 200,
                                "loss_function": "MAPE",
                                "early_stopping_rounds": 10}

        # Сюда будем складывать все преобразованные датафреймы
        dfs = []

        for column in self.COLUMNS:
            path = f"{column}.npy"
            data = np.load(path)
            if self.ELEVATION == column:
                df = self.hour_array_to_df(data, -1, column)
                self.data_elevation = df
                continue

            for hour in range(self.HOURS_TRAIN):
                df = self.hour_array_to_df(data[hour], hour, column).copy()
                dfs.append(df)

        self.train = pd.concat(dfs)

    def hour_array_to_df(self, array: np.array, hour: int, column: str) -> pd.DataFrame:
        """Перевод двумерного массива в строки для создания нового датафрейма.

        Args:
            array (np.array): Двумерный массив данных.
            hour (int): Значение часа (для высотности задана -1)
            column (str): Название столбца (тип погоды или высотность рельефа).
        Returns:
            pd.DataFrame: Преобразованные данные с нужными столбцами.
        """
        df = pd.DataFrame()
        dfs = []

        for y in range(self.SIZE_Y):
            values = array[y]
            df[column] = values
            df["y"] = y
            df["x"] = list(range(self.SIZE_X))
            df["hour"] = hour
            df["type_weather"] = column
            dfs.append(df.copy())
        return pd.concat(dfs)

    def get_result(self):
        """Вызываем необходимую модель для обучения и предсказания. Выгружаем файл с предсказаниями."""

        result = pd.DataFrame()

        if self.type_model == "base_model":
            result = self.base_model()

        if self.type_model == "catboost_model":
            result = self.catboost_model()

        if self.type_model == "mix_model":
            result_base = self.base_model()
            result_catboost = self.catboost_model()

            result = result_base[self.COLUMNS_PREDICT_BASE]
            result[self.COLUMNS_PREDICT_CATBOOST] = result_catboost[self.COLUMNS_PREDICT_CATBOOST]

        result = result[self.TYPES_WEATHER]
        print(f"Worked model: {self.type_model}")
        print(f"Shape predict: {result.shape}. Must have to (4500, 6).")
        result.to_csv(f"Predict_{self.type_model}.csv", index_label="ID")

    def base_model(self) -> pd.DataFrame:
        """Запускаем обучение и предсказание улучшенной базовой модели.

        Returns:
           pd.DataFrame: Данные с предсказаниями.
        """
        coefs = {"coef": [], "x": [], "y": [], "type_weather": []}

        start_train_hour = 0
        finish_train_hour = self.HOURS_IN_DAY - self.HOURS_TEST

        start_test_hour = self.HOURS_IN_DAY
        finish_test_hour = self.HOURS_TRAIN

        # Получаем коэффициенты для каждой координаты
        print("Start fitting base_model.")
        for type_weather in tqdm(self.TYPES_WEATHER):
            type_weather_df = self.train[self.train["type_weather"] == type_weather]

            for x in range(self.SIZE_X):
                for y in range(self.SIZE_Y):
                    train = type_weather_df[(type_weather_df.hour > start_train_hour) &
                                            (type_weather_df.hour < finish_train_hour)]
                    test = type_weather_df[(type_weather_df.hour > start_test_hour) &
                                           (type_weather_df.hour < finish_test_hour)]

                    train = train[(train.x == x) & (train.y == y)]
                    test = test[(test.x == x) & (test.y == y)]

                    coefs["coef"].append((test[type_weather] / train[type_weather]).mean())
                    coefs["x"].append(x)
                    coefs["y"].append(y)
                    coefs["type_weather"].append(type_weather)

        df_coef = pd.DataFrame(coefs)
        df_coef = df_coef.replace([np.inf, -np.inf], 1)
        print(f"Shape df_coef: {df_coef.shape}")

        # С помощью коэффициентов предсказываем значения параметров погоды
        df_merge = self.train.merge(df_coef, how="left", on=["type_weather", "x", "y"])
        df_merge["new_value"] = df_merge.apply(lambda row: row[row["type_weather"]] * row["coef"], axis=1)

        dfs_merge = []
        for type_weather in self.TYPES_WEATHER:
            type_weather_df = df_merge[df_merge["type_weather"] == type_weather].copy()
            type_weather_df[type_weather] = type_weather_df["new_value"]
            dfs_merge.append(type_weather_df)
        df_merge = pd.concat(dfs_merge)

        print("Start predict base_model.")
        dfs_predict = []
        start_time = self.HOURS_IN_DAY - self.HOURS_TEST
        finish_time = self.HOURS_IN_DAY
        for hour in range(start_time, finish_time):
            df_hour = df_merge.query(f"hour == {hour}").copy()
            dfs_predict.append(df_hour)

        df_predict = pd.concat(dfs_predict)
        print("Finish predict base_model.\n")
        return self.prepare_predict(df_predict)

    def catboost_model(self) -> pd.DataFrame:
        """Запускаем обучение и предсказание catboost.

        Returns:
           pd.DataFrame: Данные с предсказаниями.
        """
        print("Start prepare train data for catboost_model.")
        data_df = self.get_data()
        print(f"Shape train data: {data_df.shape}.")

        # Выкачиваем подобранные параметры learning_rate для каждой модели
        df_learning_rate = pd.read_csv(self.LEARNING_RATE_FILE)

        models = {"target": [], "predict_hour": [], "model": []}

        for target in self.TYPES_WEATHER:
            print(f"\nFitting model for target: {target}.")

            # Выбираем данные
            if target == self.WIND_DIR:
                train = data_df[(data_df.hour > self.HOURS_TRAIN - 5) & (data_df.wind_dir < 100)]
            else:
                train = data_df

            for predict_hour in tqdm(range(1, self.HOURS_TEST + 1)):
                dfs = []

                for x in range(self.SIZE_X):
                    for y in range(self.SIZE_Y):
                        x_y_df = train[(train.x == x) & (train.y == y)].sort_values(["hour"])
                        no_shift_df = x_y_df[["x", "y", "hour", "hour_in_day", "elevation"]]
                        x_y_df = x_y_df.shift(predict_hour)
                        x_y_df[["x", "y", "hour", "hour_in_day", "elevation"]] = no_shift_df.copy()
                        dfs.append(x_y_df.copy())

                data = pd.concat(dfs)
                x_train, x_eval, y_train, y_eval = train_test_split(data, train[target], test_size=0.3, random_state=33)

                # Получаем значение learning_rate для Catboost в зависимости от таргте и периода предсказания
                learning_rate = df_learning_rate[(df_learning_rate["target"] == target) &
                                                 (df_learning_rate["predict_hour"] == predict_hour)]
                learning_rate = learning_rate["learning_rate"].to_numpy()[0]
                self.params_catboost["learning_rate"] = learning_rate

                model = CatBoostRegressor(**self.params_catboost)
                model.fit(x_train, y_train,
                          eval_set=(x_eval, y_eval),
                          cat_features=[],
                          verbose=False)

                models["target"].append(target)
                models["predict_hour"].append(predict_hour)
                models["model"].append(model)

        models_df = pd.DataFrame(models)

        test = data_df[data_df.hour == (self.HOURS_TRAIN - 1)]
        print(f"Shape predict data: {test.shape}.\n")

        predict = pd.DataFrame()
        for target in self.TYPES_WEATHER:
            dfs_predict = []

            for predict_hour in range(1, self.HOURS_TEST + 1):
                x = test.copy()
                x["hour"] = self.HOURS_TRAIN + predict_hour
                x["hour_in_day"] = x.hour - 24

                model = models_df[(models_df["target"] == target) & (models_df["predict_hour"] == predict_hour)]
                model = model["model"].values[0]

                df = pd.DataFrame()
                df[target] = model.predict(x)
                df["hour"] = predict_hour
                df["x"] = test["x"].to_numpy()
                df["y"] = test["y"].to_numpy()
                dfs_predict.append(df.copy())

            new_target_predict = pd.concat(dfs_predict.copy())
            if predict.empty:
                predict = new_target_predict
            else:
                predict = predict.merge(new_target_predict, how="left", on=["x", "y", "hour"])

        return self.prepare_predict(predict)

    def get_data(self) -> pd.DataFrame:
        """Возвращает датафрейм в удобном формате для обучения.

        Returns:
            pd.DataFrame: Данные для обучения.
        """
        train_df = pd.DataFrame()

        for type_weather in tqdm(self.TYPES_WEATHER):
            df = self.train[self.train["type_weather"] == type_weather][[type_weather, "y", "x", "hour"]]
            overall_new_train_df = pd.DataFrame()

            for delta_hour in range(1, self.HOURS_TEST + 1):
                dfs = []
                delta_column = f"{type_weather}{delta_hour}"

                for x in range(self.SIZE_X):
                    for y in range(self.SIZE_Y):
                        x_y_df = df[(df.x == x) & (df.y == y)].sort_values(["hour"])
                        hours = x_y_df.hour
                        x_y_df[delta_column] = x_y_df[type_weather].diff(delta_hour)
                        x_y_df["hour"] = hours
                        dfs.append(x_y_df.copy())

                new_train_df = pd.concat(dfs)
                if overall_new_train_df.empty:
                    overall_new_train_df = new_train_df.copy()
                    continue

                overall_new_train_df = overall_new_train_df.merge(new_train_df, how="left",
                                                                  on=["x", "y", "hour", type_weather])

            if train_df.empty:
                train_df = overall_new_train_df.copy()
                continue

            train_df = train_df.merge(overall_new_train_df, how="left", on=["x", "y", "hour"])

        func_hour_in_day = (lambda row: row["hour"] if row["hour"] < 24 else row["hour"] - 24)
        train_df["hour_in_day"] = train_df.apply(func_hour_in_day, axis=1).astype(int)

        elevation_merge = self.data_elevation[["elevation", "x", "y"]]
        train_df = train_df.merge(elevation_merge, how="left", on=["x", "y"])

        return train_df

    def prepare_predict(self, predict: pd.DataFrame) -> pd.DataFrame:
        """Преобразовываем предикты в нужный формат для проверки.

        Args:
            predict (pd.DataFrame): Данные с предсказаниями из модели.
        Returns:
            pd.DataFrame: Преобразованные данные для проверки.
        """
        clear_predict = pd.DataFrame()

        for type_weather in self.TYPES_WEATHER:
            df = predict[[type_weather, "hour", "y", "x"]].dropna().copy()

            if clear_predict.empty:
                clear_predict = predict[[type_weather, "hour", "y", "x"]].dropna().copy()
                continue

            clear_predict = clear_predict.merge(df, how="left", on=["hour", "y", "x"])

        clear_predict = clear_predict.sort_values(["hour", "y", "x"], ignore_index=True)
        clear_predict = clear_predict.drop(["hour", "y", "x"], axis=1)

        clear_predict["cloud_cover"] = (clear_predict.cloud_cover * self.CLOUD_COVER_KOEF).clip(0).to_numpy()
        clear_predict["wind_dir"] = (clear_predict.wind_dir * self.WIND_DIR_KOEF).clip(5).to_numpy()

        return clear_predict


def main() -> None:
    type_model = "mix_model"
    weather_model = WeatherModel(type_model)
    weather_model.get_result()


if __name__ == '__main__':
    main()
