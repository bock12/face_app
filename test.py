import sys
import sqlite3
import pickle
import cv2
import os
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QTabWidget, QGridLayout, QPushButton, QScrollArea,
    QDialog, QLineEdit, QTableWidget, QTableWidgetItem,
    QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QTimer

from face_recognizer import FaceRecognizerDL
from face_db import connect_db, init_db

# -- Constants --
DB_PATH = 'face_records.db'
UNKNOWN_DIR = 'unknown_faces'

# -- Initialize database --
init_db()

# -- Initialize face recognizer --
known_encodings = []
known_names = []
recognizer = FaceRecognizerDL(known_encodings, known_names, UNKNOWN_DIR)

# -----------------------
# Unknown Faces Tab
# -----------------------
class UnknownFacesTab(QWidget):
    def __init__(self, known_faces_tab):
        super().__init__()
        self.known_faces_tab = known_faces_tab
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.grid = QGridLayout()

        conn = connect_db()
        c = conn.cursor()
        c.execute("SELECT id, image_path, encoding FROM unknown_faces ORDER BY date_detected DESC")
        self.faces = c.fetchall()
        conn.close()

        for i, (face_id, path, encoding_blob) in enumerate(self.faces):
            btn = QPushButton()
            pixmap = QPixmap(path).scaled(100, 100, Qt.KeepAspectRatio)
            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(pixmap.size())
            btn.setFixedSize(120, 120)
            btn.clicked.connect(lambda _, p=path, e=encoding_blob, id=face_id: self.show_face_detail(p, e, id))
            self.grid.addWidget(btn, i // 4, i % 4)

        content.setLayout(self.grid)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def show_face_detail(self, image_path, encoding_blob, face_id):
        dialog = QDialog(self)
        dialog.setWindowTitle("Promote to Known")
        layout = QVBoxLayout()

        image_label = QLabel()
        pixmap = QPixmap(image_path).scaled(250, 250, Qt.KeepAspectRatio)
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)

        # Input fields
        name_input = QLineEdit()
        name_input.setPlaceholderText("Full Name")
        contact_input = QLineEdit()
        contact_input.setPlaceholderText("Contact")
        age_input = QLineEdit()
        age_input.setPlaceholderText("Age")
        gender_input = QLineEdit()
        gender_input.setPlaceholderText("Gender")
        address_input = QLineEdit()
        address_input.setPlaceholderText("Address")
        occupation_input = QLineEdit()
        occupation_input.setPlaceholderText("Occupation")

        for input_field in [name_input, contact_input, age_input, gender_input, address_input, occupation_input]:
            layout.addWidget(input_field)

        save_button = QPushButton("Promote to Known")
        layout.addWidget(save_button)

        def promote():
            name = name_input.text()
            contact = contact_input.text()
            age = age_input.text()
            gender = gender_input.text()
            address = address_input.text()
            occupation = occupation_input.text()

            if not name.strip():
                QMessageBox.warning(dialog, "Error", "Name is required!")
                return

            conn = connect_db()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO known_faces (name, contact, age, gender, address, occupation, image_path, encoding, date_added)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, contact, age, gender, address, occupation, image_path, encoding_blob,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            cursor.execute("DELETE FROM unknown_faces WHERE id = ?", (face_id,))
            conn.commit()
            conn.close()

            QMessageBox.information(dialog, "Success", "Promoted to known faces!")
            

            self.refresh_unknown_faces_tab()
            self.known_faces_tab.load_known_faces()
            dialog.accept()

        save_button.clicked.connect(promote)
        dialog.setLayout(layout)
        dialog.exec_()


    def refresh_unknown_faces_tab(self):
    # Clear the current grid layout
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Reload faces from database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, image_path, encoding FROM unknown_faces ORDER BY date_detected DESC")
        self.faces = c.fetchall()
        conn.close()

        for i, (face_id, path, encoding_blob) in enumerate(self.faces):
            btn = QPushButton()
            pixmap = QPixmap(path).scaled(100, 100, Qt.KeepAspectRatio)
            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(pixmap.size())
            btn.setFixedSize(120, 120)
            btn.clicked.connect(lambda _, p=path, e=encoding_blob, id=face_id: self.show_face_detail(p, e, id))
            self.grid.addWidget(btn, i // 4, i % 4)


    

# -----------------------
# Live Feed Tab
# -----------------------
class LiveFeedTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.feed_label = QLabel("Live camera feed")
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.feed_label)

        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Run face recognition
        frame = recognizer.detect_and_recognize(frame, self.send_alert_gui)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.feed_label.setPixmap(pixmap)

    def send_alert_gui(self, name, image_path):
        print(f"[ALERT] {name} detected! Image saved at {image_path}")

    def closeEvent(self, event):
        self.cap.release()

# -----------------------
# Known Faces Tab
# -----------------------
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
        self.table.setHorizontalHeaderLabels(["Name", "Contact", "Occupation", "Photo"])
        self.table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(self.search_bar)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def load_known_faces(self):
        self.cursor.execute("SELECT id, name, contact, occupation, image_path FROM known_faces")
        self.known_faces = self.cursor.fetchall()
        self.display_faces(self.known_faces)

    def search_faces(self, query):
        filtered = [face for face in self.known_faces if query.lower() in face[1].lower()]
        self.display_faces(filtered)

    def display_faces(self, faces):
        self.table.setRowCount(0)
        for row_index, (face_id, name, contact, occupation, image_path) in enumerate(faces):

            self.table.insertRow(row_index)

            self.table.setItem(row_index, 0, QTableWidgetItem(name))
            self.table.setItem(row_index, 1, QTableWidgetItem(contact))
            self.table.setItem(row_index, 2, QTableWidgetItem(occupation))

            image_label = QLabel()

            if image_path and os.path.exists(image_path):
                pixmap = QPixmap(image_path).scaled(60, 60, Qt.KeepAspectRatio)
            else:
                pixmap = QPixmap('placeholder.png').scaled(60, 60, Qt.KeepAspectRatio)
            image_label.setPixmap(pixmap)

            image_label.mousePressEvent = lambda event, fid=face_id: self.open_edit_dialog(fid)
            self.table.setCellWidget(row_index, 3, image_label)

            details_button = QPushButton("View")
            promote_button = QPushButton("Promote")

            # Optional: Connect button slots for details/promote
            self.table.setCellWidget(row_index, 2, details_button)
            self.table.setCellWidget(row_index, 3, promote_button)

    def open_edit_dialog(self, face_id):
        conn = connect_db()
        c = conn.cursor()
        c.execute("SELECT name, contact, occupation, image_path FROM known_faces WHERE id = ?", (face_id,))
        result = c.fetchone()
        conn.close()

        if not result:
            QMessageBox.warning(self, "Error", "Face not found!")
            return

        name, contact, occupation, image_path = result

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Known Face")
        layout = QVBoxLayout()

        name_input = QLineEdit(name)
        contact_input = QLineEdit(contact)
        occupation_input = QLineEdit(occupation)

        layout.addWidget(QLabel("Name"))
        layout.addWidget(name_input)
        layout.addWidget(QLabel("Contact"))
        layout.addWidget(contact_input)
        layout.addWidget(QLabel("Occupation"))
        layout.addWidget(occupation_input)

        button_layout = QVBoxLayout()

        save_button = QPushButton("Save Changes")
        delete_button = QPushButton("Delete Face")

        button_layout.addWidget(save_button)
        button_layout.addWidget(delete_button)
        layout.addLayout(button_layout)

        def save_changes():
            new_name = name_input.text()
            new_contact = contact_input.text()
            new_occupation = occupation_input.text()

            conn = connect_db()
            c = conn.cursor()
            c.execute("""
                UPDATE known_faces
                SET name = ?, contact = ?, occupation = ?
                WHERE id = ?
            """, (new_name, new_contact, new_occupation, face_id))
            conn.commit()
            conn.close()

            QMessageBox.information(dialog, "Success", "Face information updated!")
            self.load_known_faces()
            dialog.accept()

        def delete_face():
            confirm = QMessageBox.question(dialog, "Confirm Delete", "Are you sure you want to delete this face?", 
                                        QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                conn = connect_db()
                c = conn.cursor()
                c.execute("DELETE FROM known_faces WHERE id = ?", (face_id,))
                conn.commit()
                conn.close()

                # Optionally also delete the image file from disk
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                except Exception as e:
                    print(f"Error deleting image: {e}")

                QMessageBox.information(dialog, "Deleted", "Face deleted successfully!")
                self.load_known_faces()
                dialog.accept()

        save_button.clicked.connect(save_changes)
        delete_button.clicked.connect(delete_face)

        dialog.setLayout(layout)
        dialog.exec_()



# -----------------------
# Main Window
# -----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Security GUI")
        self.setGeometry(100, 100, 1000, 700)

        tabs = QTabWidget()
        tabs.addTab(LiveFeedTab(), "Live Feed")
        
        self.known_faces_tab = KnownFacesTab(connect_db())
        tabs.addTab(UnknownFacesTab(self.known_faces_tab), "Unknown Faces")
        tabs.addTab(self.known_faces_tab, "Known Faces")

        self.setCentralWidget(tabs)

# -----------------------
# Run the App
# -----------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
