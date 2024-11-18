int ff_request_frame(AVFilterLink *link)

{

    int ret = -1;

    FF_TPRINTF_START(NULL, request_frame); ff_tlog_link(NULL, link, 1);



    if (link->closed)

        return AVERROR_EOF;

    av_assert0(!link->frame_requested);

    link->frame_requested = 1;

    while (link->frame_requested) {

        if (link->srcpad->request_frame)

            ret = link->srcpad->request_frame(link);

        else if (link->src->inputs[0])

            ret = ff_request_frame(link->src->inputs[0]);

        if (ret == AVERROR_EOF && link->partial_buf) {

            AVFrame *pbuf = link->partial_buf;

            link->partial_buf = NULL;

            ret = ff_filter_frame_framed(link, pbuf);

        }

        if (ret < 0) {

            link->frame_requested = 0;

            if (ret == AVERROR_EOF)

                link->closed = 1;

        } else {

            av_assert0(!link->frame_requested ||

                       link->flags & FF_LINK_FLAG_REQUEST_LOOP);

        }

    }

    return ret;

}
