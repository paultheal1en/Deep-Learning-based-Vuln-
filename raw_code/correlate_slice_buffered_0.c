static void correlate_slice_buffered(SnowContext *s, slice_buffer * sb, SubBand *b, DWTELEM *src, int stride, int inverse, int use_median, int start_y, int end_y){

    const int w= b->width;

    int x,y;



//    START_TIMER



    DWTELEM * line;

    DWTELEM * prev;



    if (start_y != 0)

        line = slice_buffer_get_line(sb, ((start_y - 1) * b->stride_line) + b->buf_y_offset) + b->buf_x_offset;



    for(y=start_y; y<end_y; y++){

        prev = line;

//        line = slice_buffer_get_line_from_address(sb, src + (y * stride));

        line = slice_buffer_get_line(sb, (y * b->stride_line) + b->buf_y_offset) + b->buf_x_offset;

        for(x=0; x<w; x++){

            if(x){

                if(use_median){

                    if(y && x+1<w) line[x] += mid_pred(line[x - 1], prev[x], prev[x + 1]);

                    else  line[x] += line[x - 1];

                }else{

                    if(y) line[x] += mid_pred(line[x - 1], prev[x], line[x - 1] + prev[x] - prev[x - 1]);

                    else  line[x] += line[x - 1];

                }

            }else{

                if(y) line[x] += prev[x];

            }

        }

    }



//    STOP_TIMER("correlate")

}
