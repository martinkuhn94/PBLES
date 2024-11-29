import pandas as pd


def build_attribute_model(df):
    """Build the attribute model from the DataFrame with event name prefixes."""

    base_model = pd.DataFrame(columns=["Current State", "Next State"])
    # TODO die statischen Sachen würde ich wieder auslagern, evtl wäre es auch eine Idee diese in eine große Config zu schieben, da sis sich ja sehr wiederholen
    df = df.drop(columns=["case:concept:name"])
    column_names = df.columns
    event_names = df["concept:name"].unique()
    # TODO ich glaube es ist effizienter in eine Liste zu schreiben und diese dann im gesamten an das DF zu hängen, concat ist sehr rechenintensiv
    for event_name in event_names:
        for i in range(len(column_names) - 1):
            current_state = f"{event_name}=={column_names[i]}"
            next_state = f"{event_name}=={column_names[i + 1]}"
            new_row = pd.DataFrame({"Current State": [current_state], "Next State": [next_state]})
            base_model = pd.concat([base_model, new_row], ignore_index=True)

        start_row = pd.DataFrame(
            {"Current State": ["START==START"], "Next State": [f"{event_name}=={column_names[0]}"]}
        )
        base_model = pd.concat([start_row, base_model], ignore_index=True)

        end_row = pd.DataFrame({"Current State": [f"{event_name}=={column_names[-1]}"], "Next State": ["END==END"]})
        base_model = pd.concat([base_model, end_row], ignore_index=True)

    for event_name in event_names:
        for event_name_2 in event_names:
            end_row = pd.DataFrame(
                {
                    "Current State": [f"{event_name}=={column_names[-1]}"],
                    "Next State": [f"{event_name_2}==concept:name"],
                }
            )
            base_model = pd.concat([base_model, end_row], ignore_index=True)

    return base_model
