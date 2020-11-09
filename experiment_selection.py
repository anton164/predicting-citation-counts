def experiment_selection_page():
    import pandas as pd
    import streamlit as st
    from experiments import experiment_map
    from data_tools import st_saved_dataset_selector, load_dataset

    #############
    # Select Experiment Type
    #############

    selected_dataset = st_saved_dataset_selector()

    if selected_dataset != "None":
        df = pd.read_csv(selected_dataset, index_col="PaperId")
        st.subheader("Compiled dataframe shape")
        st.write(df.shape)

        st.subheader("First 5 entries")
        st.write(df.head(5))

        EXPERIMENTS = ["Example"]

        st.header("Select Experiment")
        experiment_name = st.selectbox(
            "Select Experiment you want to run:", ["None", *experiment_map.keys()]
        )

        if experiment_name in experiment_map:
            Experiment = experiment_map[experiment_name]

            experiment = Experiment(df)
            experiment.run()


if __name__ == "__main__":
    experiment_selection_page()
