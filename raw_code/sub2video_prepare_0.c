static int sub2video_prepare(InputStream *ist, InputFilter *ifilter)

{

    AVFormatContext *avf = input_files[ist->file_index]->ctx;

    int i, w, h;



    /* Compute the size of the canvas for the subtitles stream.

       If the subtitles codecpar has set a size, use it. Otherwise use the

       maximum dimensions of the video streams in the same file. */

    w = ifilter->width;

    h = ifilter->height;

    if (!(w && h)) {

        for (i = 0; i < avf->nb_streams; i++) {

            if (avf->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {

                w = FFMAX(w, avf->streams[i]->codecpar->width);

                h = FFMAX(h, avf->streams[i]->codecpar->height);

            }

        }

        if (!(w && h)) {

            w = FFMAX(w, 720);

            h = FFMAX(h, 576);

        }

        av_log(avf, AV_LOG_INFO, "sub2video: using %dx%d canvas\n", w, h);

    }

    ist->sub2video.w = ist->resample_width  = w;

    ist->sub2video.h = ist->resample_height = h;



    /* rectangles are AV_PIX_FMT_PAL8, but we have no guarantee that the

       palettes for all rectangles are identical or compatible */

    ist->resample_pix_fmt = ifilter->format = AV_PIX_FMT_RGB32;



    ist->sub2video.frame = av_frame_alloc();

    if (!ist->sub2video.frame)

        return AVERROR(ENOMEM);

    ist->sub2video.last_pts = INT64_MIN;

    return 0;

}
