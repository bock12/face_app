import sys
import cv2
import sqlite3
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QTabWidget, QFileDialog, QLineEdit, QTableWidget, QTableWidgetItem
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import face_recognition
from face_recognizer import FaceRecognizerDL
from face_db import connect_db, init_db


class KnownFacesTab(QWidget):
    def __init__(self, db_connection):
        super().__init__()
        self.db_connection = db_connection
        self.cursor = self.db_connection.cursor()
        self.setup_ui()
        self.load_known_faces()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search by name...")
        self.search_bar.textChanged.connect(self.search_faces)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Name", "Image", "Details", "Promote"])
        self.table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(self.search_bar)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def load_known_faces(self):
        self.cursor.execute("SELECT id, name, image_path FROM known_faces")
        self.known_faces = self.cursor.fetchall()
        self.display_faces(self.known_faces)

    def search_faces(self, query):
        filtered = [face for face in self.known_faces if query.lower() in face[1].lower()]
        self.display_faces(filtered)

    def display_faces(self, faces):
        self.table.setRowCount(0)
        for row_index, (face_id, name, image_path) in enumerate(faces):
            self.table.insertRow(row_index)
            self.table.setItem(row_index, 0, QTableWidgetItem(name))

            image_label = QLabel()
            pixmap = QPixmap(image_path)
            image_label.setPixmap(pixmap.scaled(60, 60, Qt.KeepAspectRatio))
            self.table.setCellWidget(row_index, 1, image_label)

            details_button = QPushButton("View")
            promote_button = QPushButton("Promote")

            self.table.setCellWidget(row_index, 2, details_button)
            self.table.setCellWidget(row_index, 3, promote_button)


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Security System")
        self.resize(1000, 700)

        self.db_connection = init_db()
        self.tabs = QTabWidget()

        self.known_faces_tab = KnownFacesTab(connect_db())
        self.tabs.addTab(self.known_faces_tab, "Known Faces")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
