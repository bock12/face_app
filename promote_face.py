import sys, os, cv2, pickle
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from face_db import connect_db, init_db
import face_recognition
from datetime import datetime

class PromoteGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Promote Unknown Face to Known")
        self.setGeometry(100, 100, 400, 500)
        self.selected_id = None
        self.selected_img_path = None

        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.display_selected_image)

        self.image_label = QLabel("Select an image")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name to promote as...")

        self.promote_btn = QPushButton("Promote")
        self.promote_btn.clicked.connect(self.promote)

        layout = QVBoxLayout()
        layout.addWidget(self.image_list)
        layout.addWidget(self.image_label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.promote_btn)
        self.setLayout(layout)

        self.load_unknowns()

    def load_unknowns(self):
        self.image_list.clear()
        conn = connect_db()
        c = conn.cursor()
        c.execute("SELECT id, image_path FROM unknown_faces")
        self.unknown_faces = c.fetchall()
        conn.close()

        for row in self.unknown_faces:
            self.image_list.addItem(f"{row[0]} - {os.path.basename(row[1])}")

    def display_selected_image(self, item):
        index = self.image_list.currentRow()
        self.selected_id, self.selected_img_path = self.unknown_faces[index]

        image = cv2.imread(self.selected_img_path)
        if image is None:
            self.image_label.setText("Image not found.")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        qt_image = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(300, 300, Qt.KeepAspectRatio))

    def promote(self):
        name = self.name_input.text().strip()
        if not self.selected_id or not name:
            QMessageBox.warning(self, "Missing Info", "Select image and enter a name.")
            return

        # Load encoding
        image = face_recognition.load_image_file(self.selected_img_path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            QMessageBox.warning(self, "Face Error", "No face encoding found in image.")
            return

        encoding_blob = pickle.dumps(encodings[0])
        conn = connect_db()
        c = conn.cursor()

        c.execute("INSERT INTO known_faces (name, encoding, date_added) VALUES (?, ?, ?)",
                  (name, encoding_blob, datetime.now().isoformat()))
        c.execute("DELETE FROM unknown_faces WHERE id = ?", (self.selected_id,))
        conn.commit()
        conn.close()

        try:
            os.remove(self.selected_img_path)
        except:
            pass

        QMessageBox.information(self, "Success", f"{name} promoted!")
        self.name_input.clear()
        self.image_label.clear()
        self.load_unknowns()

if __name__ == "__main__":
    from face_db import init_db
    init_db()
    app = QApplication(sys.argv)
    window = PromoteGUI()
    window.show()
    sys.exit(app.exec_())
