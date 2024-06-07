from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
import threading
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

# Hard-coded paths
MODEL_PATH = "C:/Users/HP/mental_health_classifier/model"
EXCEL_FILE_PATH = "C:/Users/HP/mental_health_classifier/data/messages.xlsx"
TRAINED_MODEL_PATH = "C:/Users/HP/mental_health_classifier/output/personalized_model"
LABELED_DATA_FILE = "labeled_data.csv"

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(TRAINED_MODEL_PATH, exist_ok=True)

# Download the tokenizer and model if not already present
if not os.path.exists(os.path.join(MODEL_PATH, 'config.json')):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.save_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.save_pretrained(MODEL_PATH)

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Preprocess text function
def preprocess_text(text):
    return text.lower()

class TooltippedButton(Button):
    def __init__(self, tooltip_text="", **kwargs):
        super(TooltippedButton, self).__init__(**kwargs)
        self.tooltip_text = tooltip_text
        self.tooltip = None
        self.bind(on_enter=self.show_tooltip)
        self.bind(on_leave=self.hide_tooltip)
        self.bind(on_press=self.on_press_event)
        self.bind(on_release=self.on_release_event)

    def show_tooltip(self, *args):
        if not self.tooltip:
            self.tooltip = Label(text=self.tooltip_text, size_hint=(None, None))
            self.tooltip.size = self.tooltip.texture_size
            self.tooltip.pos = (self.right, self.top)
            self.parent.add_widget(self.tooltip)

    def hide_tooltip(self, *args):
        if self.tooltip:
            self.parent.remove_widget(self.tooltip)
            self.tooltip = None

    def on_press_event(self, *args):
        self.background_color = (1, 0, 0, 1)  # Red color on press

    def on_release_event(self, *args):
        self.background_color = (1, 1, 1, 1)  # Default color on release

class MentalHealthClassifierApp(App):
    def build(self):
        self.title = "Mental Health Classifier"
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        self.training_data = []
        self.df = None
        self.prediction_results = []

        self.main_layout = GridLayout(cols=1, padding=10, spacing=10)

        self.upload_button = TooltippedButton(text="Upload Excel File", tooltip_text="Upload an Excel file containing messages", size_hint=(1, 0.1))
        self.upload_button.bind(on_press=self.show_filechooser)
        self.main_layout.add_widget(self.upload_button)

        self.message_label = Label(text="Label Message:", size_hint=(1, 0.1))
        self.main_layout.add_widget(self.message_label)

        self.slider_layout = GridLayout(cols=2, size_hint=(1, 0.5))
        self.sliders = {}
        conditions = ['Depression', 'Anxiety', 'Fatigue', 'Cognitive Impairment', 'Other Health Issues']
        for condition in conditions:
            condition_label = Label(text=f"{condition}:", size_hint=(0.5, 0.1))
            self.slider_layout.add_widget(condition_label)
            slider = Slider(min=-1, max=1, value=0, step=0.1, size_hint=(0.5, 0.1))
            slider.bind(value=self.on_slider_value_change)
            self.slider_layout.add_widget(slider)
            self.sliders[condition] = slider

        self.not_relevant_checkbox = CheckBox(size_hint=(0.1, 0.1))
        self.not_relevant_label = Label(text="Not Relevant", size_hint=(0.9, 0.1))
        self.slider_layout.add_widget(self.not_relevant_label)
        self.slider_layout.add_widget(self.not_relevant_checkbox)

        self.main_layout.add_widget(self.slider_layout)

        self.add_label_button = TooltippedButton(text="Add Label", tooltip_text="Add label to the current message", size_hint=(1, 0.1))
        self.add_label_button.bind(on_press=self.add_label)
        self.main_layout.add_widget(self.add_label_button)

        self.train_button = TooltippedButton(text="Train Model", tooltip_text="Train the model with labeled data", size_hint=(1, 0.1))
        self.train_button.bind(on_press=self.train_model)
        self.main_layout.add_widget(self.train_button)

        self.input_text = TextInput(hint_text="Enter message for prediction", size_hint=(1, 0.1))
        self.main_layout.add_widget(self.input_text)

        self.predict_button = TooltippedButton(text="Predict", tooltip_text="Predict the label of the input message", size_hint=(1, 0.1))
        self.predict_button.bind(on_press=self.predict)
        self.main_layout.add_widget(self.predict_button)

        self.result_label = Label(text="", size_hint=(1, 0.1))
        self.main_layout.add_widget(self.result_label)

        self.view_results_button = TooltippedButton(text="View Results", tooltip_text="View prediction results", size_hint=(1, 0.1))
        self.view_results_button.bind(on_press=self.view_results)
        self.main_layout.add_widget(self.view_results_button)

        self.export_results_button = TooltippedButton(text="Export Results", tooltip_text="Export prediction results to Excel", size_hint=(1, 0.1))
        self.export_results_button.bind(on_press=self.export_results)
        self.main_layout.add_widget(self.export_results_button)

        self.labeling_info_label = Label(text="Please label at least 50 messages to start training.", size_hint=(1, 0.1))
        self.main_layout.add_widget(self.labeling_info_label)
        self.labeled_count = 0

        self.scroll_view = ScrollView(size_hint=(1, 0.2))
        self.main_layout.add_widget(self.scroll_view)
        self.message_content_label = Label(text="", size_hint=(1, None), halign="left", valign="top")
        self.message_content_label.bind(size=self.message_content_label.setter('text_size'))
        self.scroll_view.add_widget(self.message_content_label)

        self.load_labeled_data()

        return self.main_layout

    def load_labeled_data(self):
        if os.path.exists(LABELED_DATA_FILE):
            self.training_data = pd.read_csv(LABELED_DATA_FILE).to_dict('records')
            self.labeled_count = len(self.training_data)
            self.labeling_info_label.text = f"{self.labeled_count} messages labeled. Please label at least 50 messages to start training."
        else:
            self.training_data = []
            self.labeled_count = 0

    def save_labeled_data(self):
        pd.DataFrame(self.training_data).drop_duplicates(subset='Text').to_csv(LABELED_DATA_FILE, index=False)

    def on_slider_value_change(self, instance, value):
        if value < 0:
            instance.background_color = (1, 0, 0, 1)  # Red for negative values
        elif value > 0:
            instance.background_color = (0, 1, 0, 1)  # Green for positive values
        else:
            instance.background_color = (0, 0, 1, 1)  # Blue for neutral values

    def show_filechooser(self, instance):
        content = BoxLayout(orientation='vertical')
        filechooser = FileChooserListView(filters=['*.xlsx'], path=os.path.dirname(EXCEL_FILE_PATH))
        content.add_widget(filechooser)

        button_layout = BoxLayout(size_hint_y=None, height=50)
        select_button = Button(text="Select", size_hint_x=None, width=100)
        cancel_button = Button(text="Cancel", size_hint_x=None, width=100)
        button_layout.add_widget(select_button)
        button_layout.add_widget(cancel_button)
        content.add_widget(button_layout)

        popup = Popup(title='Select Excel File', content=content, size_hint=(0.9, 0.9))

        def on_selection(instance, value):
            if value:
                selected_file = value[0]
                threading.Thread(target=self.load_excel_file, args=(selected_file, popup)).start()

        filechooser.bind(on_selection=on_selection)
        select_button.bind(on_press=lambda x: on_selection(filechooser, filechooser.selection))
        cancel_button.bind(on_press=popup.dismiss)
        popup.open()

    def load_excel_file(self, selected_file, popup):
        try:
            self.df = pd.read_excel(selected_file)
            if "Message" not in self.df.columns:
                Clock.schedule_once(lambda dt: self.show_popup("Error", "Excel file must contain a 'Message' column."), 0)
            else:
                Clock.schedule_once(lambda dt: self.label_message(), 0)
            Clock.schedule_once(lambda dt: popup.dismiss(), 0)
        except Exception as e:
            Clock.schedule_once(lambda dt: self.show_popup("Error", f"Failed to load file: {str(e)}"), 0)
            Clock.schedule_once(lambda dt: popup.dismiss(), 0)

    def show_popup(self, title, message):
        popup_content = BoxLayout(orientation='vertical', padding=10)
        popup_content.add_widget(Label(text=message))
        close_button = Button(text="Close")
        popup_content.add_widget(close_button)
        popup = Popup(title=title, content=popup_content, size_hint=(0.7, 0.7))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

    def label_message(self):
        if self.df is not None:
            self.labeled_count = len(self.training_data)
            self.labeling_info_label.text = f"{self.labeled_count} messages labeled. Please label at least 50 messages to start training."
            
            remaining_messages = self.df[~self.df['Message'].isin([d['Text'] for d in self.training_data])]
            if not remaining_messages.empty:
                message_row = remaining_messages.sample(n=1).iloc[0]
                self.current_message = message_row['Message']
                self.current_sender = message_row.get('Sender', 'Unknown')
                self.current_datetime = message_row.get('Date/Time', 'Unknown')  # Adjusted to read Date/Time
                self.message_content_label.text = f"Sender: {self.current_sender}\nDate/Time: {self.current_datetime}\n\n{self.current_message}"
            else:
                self.message_label.text = "All messages have been labeled."

    def add_label(self, instance):
        label_values = {condition: self.sliders[condition].value for condition in self.sliders}
        label_values['Not Relevant'] = self.not_relevant_checkbox.active
        self.training_data.append({
            'Text': preprocess_text(self.current_message), 
            'Sender': self.current_sender,
            'DateTime': self.current_datetime,
            'Labels': label_values
        })
        self.show_popup("Info", "Label added to training data.")
        self.save_labeled_data()
        self.reset_sliders_and_checkbox()
        Clock.schedule_once(lambda dt: self.label_message(), 0)

    def reset_sliders_and_checkbox(self):
        for slider in self.sliders.values():
            slider.value = 0
        self.not_relevant_checkbox.active = False

    def train_model(self, instance):
        if self.labeled_count < 50:
            self.show_popup("Error", "Please label at least 50 messages before training the model.")
            return
        self.train_button.disabled = True
        self.progress_bar = ProgressBar(max=100)
        self.main_layout.add_widget(self.progress_bar)
        threading.Thread(target=self._train_model).start()

    def _train_model(self):
        try:
            df = pd.DataFrame(self.training_data)
            labels = {condition: [] for condition in self.sliders}
            labels['Not Relevant'] = []
            for row in df.itertuples():
                for condition in self.sliders:
                    labels[condition].append(row.Labels[condition])
                labels['Not Relevant'].append(row.Labels['Not Relevant'])
            dataset = Dataset.from_pandas(pd.DataFrame({'Text': df['Text'], **labels}))

            training_args = TrainingArguments(
                output_dir=TRAINED_MODEL_PATH,
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(TRAINED_MODEL_PATH, 'logs'),
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch"
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=dataset
            )

            trainer.train()
            self.model.save_pretrained(TRAINED_MODEL_PATH)
            Clock.schedule_once(lambda dt: self.show_popup("Info", "Model training completed."), 0)
        except Exception as e:
            Clock.schedule_once(lambda dt: self.show_popup("Error", f"Training failed: {str(e)}"), 0)
        finally:
            Clock.schedule_once(lambda dt: self.enable_train_button(), 0)
            Clock.schedule_once(lambda dt: self.main_layout.remove_widget(self.progress_bar), 0)

    def enable_train_button(self):
        self.train_button.disabled = False

    def predict(self, instance):
        input_message = self.input_text.text
        if not input_message:
            self.show_popup("Error", "Please enter a message to predict.")
            return

        inputs = self.tokenizer(preprocess_text(input_message), return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = {condition: torch.argmax(predictions[i]).item() for i, condition in enumerate(self.sliders)}

        result = {'Message': input_message, 'Predicted_Labels': predicted_labels}
        self.prediction_results.append(result)
        self.result_label.text = f"Predicted Labels: {predicted_labels}"
        self.show_feedback_ui(result)

    def show_feedback_ui(self, result):
        layout = BoxLayout(orientation='vertical', padding=10)
        layout.add_widget(Label(text="Was the prediction accurate?"))
        for condition, predicted_label in result['Predicted_Labels'].items():
            layout.add_widget(Label(text=f"{condition}: {predicted_label}"))

        correct_button = Button(text="Correct")
        incorrect_button = Button(text="Incorrect")
        layout.add_widget(correct_button)
        layout.add_widget(incorrect_button)

        feedback_popup = Popup(title='Prediction Feedback', content=layout, size_hint=(0.7, 0.7))
        correct_button.bind(on_press=lambda x: self.handle_feedback(feedback_popup, result, True))
        incorrect_button.bind(on_press=lambda x: self.handle_feedback(feedback_popup, result, False))
        feedback_popup.open()

    def handle_feedback(self, popup, result, correct):
        if not correct:
            # Handle incorrect feedback logic
            pass
        popup.dismiss()

    def view_results(self, instance):
        if not self.prediction_results:
            self.show_popup("Error", "No prediction results available.")
            return

        layout = BoxLayout(orientation='vertical', padding=10)

        filter_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        filter_layout.add_widget(Label(text="Filter by label:"))
        label_filter = Spinner(text='All', values=['All', '-1', '0', '1'])
        filter_layout.add_widget(label_filter)
        layout.add_widget(filter_layout)

        sort_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        sort_layout.add_widget(Label(text="Sort by:"))
        sort_criteria = Spinner(text='Message', values=['Message', 'Predicted Label'])
        sort_layout.add_widget(sort_criteria)
        layout.add_widget(sort_layout)

        result_layout = BoxLayout(orientation='vertical', padding=10)
        scrollview = ScrollView()
        grid = GridLayout(cols=2, size_hint_y=None)
        grid.bind(minimum_height=grid.setter('height'))

        headers = ["Message", "Predicted Label"]
        for header in headers:
            grid.add_widget(Label(text=header, bold=True))

        def update_results(*args):
            grid.clear_widgets()
            for header in headers:
                grid.add_widget(Label(text=header, bold=True))

            filtered_results = self.prediction_results
            if label_filter.text != 'All':
                filtered_results = [r for r in self.prediction_results if str(r['Predicted_Labels']) == label_filter.text]

            if sort_criteria.text == 'Message':
                filtered_results = sorted(filtered_results, key=lambda x: x['Message'])
            elif sort_criteria.text == 'Predicted Label':
                filtered_results = sorted(filtered_results, key=lambda x: x['Predicted_Labels'])

            for result in filtered_results:
                grid.add_widget(Label(text=result['Message']))
                grid.add_widget(Label(text=str(result['Predicted_Labels'])))

        label_filter.bind(text=update_results)
        sort_criteria.bind(text=update_results)

        scrollview.add_widget(grid)
        result_layout.add_widget(scrollview)
        layout.add_widget(result_layout)

        popup = Popup(title='Prediction Results', content=layout, size_hint=(0.9, 0.9))
        popup.open()

    def export_results(self, instance):
        if not self.prediction_results:
            self.show_popup("Error", "No results to export.")
            return
        try:
            df = pd.DataFrame(self.prediction_results)
            df.to_excel(os.path.join(TRAINED_MODEL_PATH, 'prediction_results.xlsx'), index=False)
            self.show_popup("Info", "Results exported successfully.")
        except Exception as e:
            self.show_popup("Error", f"Export failed: {str(e)}")

if __name__ == '__main__':
    MentalHealthClassifierApp().run()
