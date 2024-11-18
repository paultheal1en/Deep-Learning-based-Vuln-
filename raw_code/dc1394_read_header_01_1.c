static int dc1394_read_header(AVFormatContext *c)
{
    dc1394_data* dc1394 = c->priv_data;
    dc1394camera_list_t *list;
    int res, i;
    const struct dc1394_frame_format *fmt = NULL;
    const struct dc1394_frame_rate *fps = NULL;
    if (dc1394_read_common(c, &fmt, &fps) != 0)
       return -1;
    /* Now let us prep the hardware. */
    dc1394->d = dc1394_new();
    if (dc1394_camera_enumerate(dc1394->d, &list) != DC1394_SUCCESS || !list) {
        av_log(c, AV_LOG_ERROR, "Unable to look for an IIDC camera.\n");
    if (list->num == 0) {
        av_log(c, AV_LOG_ERROR, "No cameras found.\n");
    /* FIXME: To select a specific camera I need to search in list its guid */
    dc1394->camera = dc1394_camera_new (dc1394->d, list->ids[0].guid);
    if (list->num > 1) {
        av_log(c, AV_LOG_INFO, "Working with the first camera found\n");
    /* Freeing list of cameras */
    dc1394_camera_free_list (list);
    /* Select MAX Speed possible from the cam */
    if (dc1394->camera->bmode_capable>0) {
       dc1394_video_set_operation_mode(dc1394->camera, DC1394_OPERATION_MODE_1394B);
       i = DC1394_ISO_SPEED_800;
    } else {
       i = DC1394_ISO_SPEED_400;
    for (res = DC1394_FAILURE; i >= DC1394_ISO_SPEED_MIN && res != DC1394_SUCCESS; i--) {
            res=dc1394_video_set_iso_speed(dc1394->camera, i);
    if (res != DC1394_SUCCESS) {
        av_log(c, AV_LOG_ERROR, "Couldn't set ISO Speed\n");
        goto out_camera;
    if (dc1394_video_set_mode(dc1394->camera, fmt->frame_size_id) != DC1394_SUCCESS) {
        av_log(c, AV_LOG_ERROR, "Couldn't set video format\n");
        goto out_camera;
    if (dc1394_video_set_framerate(dc1394->camera,fps->frame_rate_id) != DC1394_SUCCESS) {
        av_log(c, AV_LOG_ERROR, "Couldn't set framerate %d \n",fps->frame_rate);
        goto out_camera;
    if (dc1394_capture_setup(dc1394->camera, 10, DC1394_CAPTURE_FLAGS_DEFAULT)!=DC1394_SUCCESS) {
        av_log(c, AV_LOG_ERROR, "Cannot setup camera \n");
        goto out_camera;
    if (dc1394_video_set_transmission(dc1394->camera, DC1394_ON) !=DC1394_SUCCESS) {
        av_log(c, AV_LOG_ERROR, "Cannot start capture\n");
        goto out_camera;
    return 0;
out_camera:
    dc1394_capture_stop(dc1394->camera);
    dc1394_video_set_transmission(dc1394->camera, DC1394_OFF);
    dc1394_camera_free (dc1394->camera);
out:
    dc1394_free(dc1394->d);
    return -1;