get_pointer_coordinates(int *x, int *y, Display *dpy, AVFormatContext *s1)

{

    Window mrootwindow, childwindow;

    int dummy;



    mrootwindow = DefaultRootWindow(dpy);



    if (XQueryPointer(dpy, mrootwindow, &mrootwindow, &childwindow,

                      x, y, &dummy, &dummy, (unsigned int*)&dummy)) {

    } else {

        av_log(s1, AV_LOG_INFO, "couldn't find mouse pointer\n");

        *x = -1;

        *y = -1;

    }

}
