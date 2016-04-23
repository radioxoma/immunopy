# Immunopy

*Immunopy — realtime immunostain analysis application.*

The program acquires microscope camera video stream and performs realtime image analysis of current field of view. With colour labels and statistics (labeling index, cell count), showing as video overlay, pathologist can observe assay in *augmented* way.

Immunopy targeted to breast cancer immunohistochemical assays with nuclear markers (Ki-67, estrogen and progesterone receptors stained with DAB & hematoxylin).

**This project has been my thesis work at the microbiology & pathology departments of [Vitebsk state medical university](http://www.vsmu.by). Free access article (including fulltext PDF) published at [elibrary.ru](http://elibrary.ru/item.asp?id=25643923). Short [demo video](https://www.youtube.com/watch?v=jwfPKooYHZs) published at Youtube.**


### Резюме

Рак молочной железы по-прежнему является ведущей онкопатологией у женщин. Точная оценка экспрессии опухолевых маркеров является принципиальной для подбора химиотерапевтических препаратов, оценки прогноза и экономической эффективности лечения для конкретного пациента. Учитывая известные проблемы применения иммуногистохимических методов в рутинной патоморфологической диагностике, в настоящей работе была разработана методика, способствующая стандартизации, снижению влияния человеческого фактора и уменьшению временных затрат на документирование.

В основе метода лежит разработанное авторами программное обеспечение «Immunopy», позволяющее обрабатывать видеопоток с камеры, присоединённой к оптическому микроскопу. Анализ происходит в режиме реального времени одновременно с проведением визуальной оценки гистопрепарата. Программа создаёт видео «дополненной реальности»: поверх позитивных и негативных клеток накладываются цветные маркеры, облегчающие восприятие, а также количественная информация такая как общее число клеток, индекс позитивных клеток. Вся необходимая информация рассчитывается и отображается автоматически.
Пользователь может сохранить фотографии, экспортировать статистику для архивации или дальнейшего анализа в программах для работы с электронными таблицами таких как Microsoft Excel или LibreOffice Calc.

В настоящей работе проведён корреляционный анализ между визуальной и автоматической оценкой индекса позитивных клеток (r<sub>pearson</sub> = 0,91; r<sub>spearman</sub> = 0,8; p < 0,0001).

Immunopy является свободным программным обеспечением, исходный код программы доступен в сети Интернет под лицензией MIT. Предложенная методика может найти широкое применение в клинической практике и научной работе.

Ключевые слова: рак молочной железы; иммуногистохимия; цитометрия; автоматический анализ изображений; машинное зрение; Ki-67; рецепторы эстрогена альфа; рецепторы прогестерона.


### Abstract

Breast cancer is the most common oncologic pathology among women worldwide. Precise cancer markers assessment is crucial for treatment development, evaluation of prognosis and economic efficiency for a given patient. Taking into consideration known issues with immunohistochemical techniques in routine pathomorphological diagnosis, the new method for automatic assay analysis was developed. It reduces interobserver variation, time consumption and requires less effort for documentation.

The method is based on developed original software called "Immunopy", which allows to perform video processing from camera attached to an optical microscope. Analysis accomplishes in real time, simultaneously with visual slide assessment.
The program produces "augmented reality" video with colour markers overlay, which facilitates distinguishing between positive and negative cells. Numerical cell features such as count, labeling index displayed as well. User can save acquired photos, and export statistics in spreadsheet programs like Microsoft Excel or LibreOffice Calc.

Correlation analysis between visual and automatic assessment of labeling index (r<sub>pearson</sub> = 0.91; r<sub>spearman</sub> = 0.8; p < 0.0001) performed as well.

Immunopy is free software and source code is distributed under the terms of MIT license.
Given methods and algorithms can be found useful in clinical practice and research.

Key words: breast neoplasms, immunohistochemistry, image cytometry, computer-assisted image analysis, computer-assisted image interpretation, Ki-67 antigen, estrogen receptor alpha, progesterone receptors.


## Installation

Immunopy is written in python 2. Dependencies that not supported by [pip](https://pip.pypa.io) are listed in `setup.py` file near with "install_requires" section. The program uses OpenCL for colour deconvolution if available.

    python2 setup.py install


## Configuration

Image acquisition relies on [Micro-manager](https://www.micro-manager.org). You need to create [configuration file](https://micro-manager.org/wiki/Micro-Manager_Configuration_Guide) (e.g. `camera_demo.cfg`) with group *"System"* and preset *"Startup"*. This device configuration will be loaded as default on startup, also some camera settings can be changed during work.

It's necessary to calibrate microscope and define pixel size for used magnifications.
