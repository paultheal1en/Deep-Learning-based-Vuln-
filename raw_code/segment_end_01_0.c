static int segment_end(AVFormatContext *s, int write_trailer, int is_last)

{

    SegmentContext *seg = s->priv_data;

    AVFormatContext *oc = seg->avf;

    int ret = 0;



    av_write_frame(oc, NULL); /* Flush any buffered data (fragmented mp4) */

    if (write_trailer)

        ret = av_write_trailer(oc);



    if (ret < 0)

        av_log(s, AV_LOG_ERROR, "Failure occurred when ending segment '%s'\n",

               oc->filename);



    if (seg->list) {

        if (seg->list_size || seg->list_type == LIST_TYPE_M3U8) {

            SegmentListEntry *entry = av_mallocz(sizeof(*entry));

            if (!entry) {

                ret = AVERROR(ENOMEM);

                goto end;

            }



            /* append new element */

            memcpy(entry, &seg->cur_entry, sizeof(*entry));

            if (!seg->segment_list_entries)

                seg->segment_list_entries = seg->segment_list_entries_end = entry;

            else

                seg->segment_list_entries_end->next = entry;

            seg->segment_list_entries_end = entry;



            /* drop first item */

            if (seg->list_size && seg->segment_count > seg->list_size) {

                entry = seg->segment_list_entries;

                seg->segment_list_entries = seg->segment_list_entries->next;

                av_free(entry->filename);

                av_freep(&entry);

            }



            avio_close(seg->list_pb);

            if ((ret = segment_list_open(s)) < 0)

                goto end;

            for (entry = seg->segment_list_entries; entry; entry = entry->next)

                segment_list_print_entry(seg->list_pb, seg->list_type, entry, s);

            if (seg->list_type == LIST_TYPE_M3U8 && is_last)

                avio_printf(seg->list_pb, "#EXT-X-ENDLIST\n");

        } else {

            segment_list_print_entry(seg->list_pb, seg->list_type, &seg->cur_entry, s);

        }

        avio_flush(seg->list_pb);

    }



    av_log(s, AV_LOG_VERBOSE, "segment:'%s' count:%d ended\n",

           seg->avf->filename, seg->segment_count);

    seg->segment_count++;



end:

    avio_close(oc->pb);



    return ret;

}
