# Detección de somnolencia y distracciones al volante

Conducir es una actividad compleja que requiere percibir continuamente la situación cambiante del tráfico, evaluarla, decidir las acciones más adecuadas a realizar en cada caso y ejecutar correctamente estas acciones. Todo este proceso precisa que el conductor se encuentre en óptimas condiciones psicofísicas para que pueda llevarse a cabo adecuadamente.

Conjunto de datos:
- Dataset containing smoking and not-smoking images (smoker vs non-smoker)
- Mobile Images Dataset
- Coco Sets de imágenes en un 50%.
- Máscara de ubicación de de puntos de referencia de rostros: shape_predictor_68_face_landmarks.dat

Modelo: 
- Transferencia de aprendizaje de Yolo V5 para detección de distracciones y DLIB con shape_predictor_68_face_landmarks para detección de somnolencia.
