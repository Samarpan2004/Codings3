import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.face.LBPHFaceRecognizer; // from opencv_contrib
import org.opencv.face.Face;

import java.io.File;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

public class FaceRecognition {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) throws Exception {
        String faceDir = "faces"; // contains subfolders for each person
        List<Mat> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        List<String> labelNames = new ArrayList<>();

        int labelCounter = 0;
        for (File personFolder : new File(faceDir).listFiles()) {
            if (!personFolder.isDirectory()) continue;
            String person = personFolder.getName();
            labelNames.add(person);
            for (File imgFile : personFolder.listFiles()) {
                Mat img = Imgcodecs.imread(imgFile.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
                if (img.empty()) continue;
                Imgproc.resize(img, img, new Size(200, 200));
                images.add(img);
                labels.add(labelCounter);
            }
            labelCounter++;
        }

        MatOfInt labelsMat = new MatOfInt();
        labelsMat.fromArray(labels.stream().mapToInt(i->i).toArray());

        LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create(1,8,8,8,200);
        recognizer.train(images, labelsMat);
        System.out.println("Training done. Labels: " + labelNames);

        // Load face detector
        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");

        VideoCapture cap = new VideoCapture(0);
        if (!cap.isOpened()) { System.err.println("Cannot open camera"); return; }
        Mat frame = new Mat();
        while (true) {
            if (!cap.read(frame)) break;
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(gray, faces);
            for (Rect r : faces.toArray()) {
                Mat face = new Mat(gray, r);
                Imgproc.resize(face, face, new Size(200,200));
                int[] predicted = new int[1];
                double[] confidence = new double[1];
                recognizer.predict(face, predicted, confidence);
                String name = "Unknown";
                if (predicted[0] >=0 && predicted[0] < labelNames.size()) {
                    name = labelNames.get(predicted[0]);
                }
                Imgproc.rectangle(frame, r.tl(), r.br(), new Scalar(0,255,0), 2);
                Imgproc.putText(frame, name + String.format(" (%.1f)", confidence[0]), r.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255,0,0),2);
            }
            // Show frame using HighGui (or save/serve via Swing)
            // HighGui.imshow("Face Recognition", frame);
            // if (HighGui.waitKey(30) >= 0) break;
            // For headless example, just break after first loop
            break;
        }
        cap.release();
        System.out.println("Done.");
    }
}
