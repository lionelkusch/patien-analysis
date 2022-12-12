(define (batch-autocrop-file filename)
  (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE
                        filename filename)))
          (drawable (car (gimp-image-get-active-layer image))))

    (plug-in-autocrop RUN-NONINTERACTIVE
      image drawable)

    (gimp-file-save RUN-NONINTERACTIVE
      image drawable filename filename)
    (gimp-image-delete image))
  )