static int xv_write_trailer(AVFormatContext *s)

{

    XVContext *xv = s->priv_data;



    XShmDetach(xv->display, &xv->yuv_shminfo);

    shmdt(xv->yuv_image->data);

    XFree(xv->yuv_image);


    XCloseDisplay(xv->display);

    return 0;

}