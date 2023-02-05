//
//  ViewController.swift
//  MiDas_Test
//
//  Created by Jaehyeon Park on 2023/02/04.
//

import UIKit
import Vision
import CoreMedia

class ViewController: UIViewController {

    @IBOutlet weak var previewView: UIView!
    
    @IBOutlet weak var imageView: UIImageView!
    var videoCapture: VideoCapture!
    
    let objectDectectionModel = try! MiDaS()
    let semaphore = DispatchSemaphore(value: 1)
    
    // MARK: - Vision Properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    var isInferencing = false
    
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        setUpModel()
        setUpCamera()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        self.videoCapture.start()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        self.videoCapture.stop()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        resizePreviewLayer()
    }
    
    
    // MARK: - Setup Core ML
    func setUpModel() {
        if let visionModel = try? VNCoreMLModel(for: objectDectectionModel.model) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .centerCrop
        } else {
            fatalError("fail to create vision model")
        }
    }
    
    // MARK: - SetUp Video
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 30
        videoCapture.setUp(sessionPreset: .vga640x480) { success in
            
            if success {
                // add preview view on the layer
//                if let previewLayer = self.videoCapture.previewLayer {
//                    self.previewView.layer.addSublayer(previewLayer)
//                    //self.resizePreviewLayer()
//                }
                
                // start video preview when setup is done
                self.videoCapture.start()
            }
        }
    }
    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = previewView.bounds
    }
    
    func predictUsingVision(pixelBuffer: CVPixelBuffer) {
        guard let request = request else { fatalError() }
        // vision framework configures the input size of image following our model's input configuration automatically
        self.semaphore.wait()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }


}

// MARK: - VideoCaptureDelegate
extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
        
        // the captured image from camera is contained on pixelBuffer
        if !self.isInferencing, let pixelBuffer = pixelBuffer {
            self.isInferencing = true

            // predict!
            self.predictUsingVision(pixelBuffer: pixelBuffer)
        }
    }
    
    // MARK: - Post-processing
    func visionRequestDidComplete(request: VNRequest, error: Error?) {

        if let predictions = request.results as? [VNPixelBufferObservation],
           let pixelBuffer = predictions.first?.pixelBuffer {
            
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let uiImage = UIImage(ciImage: ciImage)
            
            DispatchQueue.main.async {
                self.imageView.image = uiImage
                self.isInferencing = false
            }
        } else {

            self.isInferencing = false
        }
        self.semaphore.signal()
    }
}

