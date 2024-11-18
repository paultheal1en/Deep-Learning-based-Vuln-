static int check_output_constraints(InputStream *ist, OutputStream *ost)

{

    OutputFile *of = output_files[ost->file_index];

    int ist_index  = input_files[ist->file_index]->ist_index + ist->st->index;



    if (ost->source_index != ist_index)

        return 0;



    if (of->start_time && ist->pts < of->start_time)

        return 0;



    if (of->recording_time != INT64_MAX &&

        av_compare_ts(ist->pts, AV_TIME_BASE_Q, of->recording_time + of->start_time,

                      (AVRational){ 1, 1000000 }) >= 0) {

        ost->is_past_recording_time = 1;

        return 0;

    }



    return 1;

}
