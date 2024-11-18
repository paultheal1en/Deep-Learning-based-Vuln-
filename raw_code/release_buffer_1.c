static void release_buffer(AVCodecContext *avctx, AVFrame *pic)

{

    int i;



    CVPixelBufferRef cv_buffer = (CVPixelBufferRef)pic->data[3];

    CVPixelBufferUnlockBaseAddress(cv_buffer, 0);

    CVPixelBufferRelease(cv_buffer);



    for (i = 0; i < 4; i++)

        pic->data[i] = NULL;

}
