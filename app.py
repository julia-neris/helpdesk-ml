import gradio as gr
from predict import predict

gr.Interface(fn=predict, inputs=gr.Textbox(label="Chamado de TI", max_lines=6),
             outputs=gr.JSON(label="Sugestão para revisão"),
             title="Helpdesk ML",
             description="Protótipo treinado com dados sintéticos. Todas as sugestões exigem revisão humana.",
             examples=[["Esqueci minha senha"], ["O dashboard tem registros duplicados"]]).launch()
