cd Presentation/Beamer

gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dNOPAUSE -dQUIET -dBATCH -dPrinted=false -sOutputFile=Presentation-compressed.pdf Presentation.pdf

gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=TRAN-THUONG-TIPE.pdf Presentation.pdf



-dPDFSETTINGS=/screen lower quality, smaller size. (72 dpi)
-dPDFSETTINGS=/ebook for better quality, but slightly larger pdfs. (150 dpi)
-dPDFSETTINGS=/prepress output similar to Acrobat Distiller "Prepress Optimized" setting (300 dpi)
-dPDFSETTINGS=/printer selects output similar to the Acrobat Distiller "Print Optimized" setting (300 dpi)
-dPDFSETTINGS=/default selects output intended to be useful across a wide variety of uses, possibly at the expense of a larger output file
