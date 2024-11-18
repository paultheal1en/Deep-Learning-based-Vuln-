void ff_release_unused_pictures(MpegEncContext *s, int remove_current)

{

    int i;



    /* release non reference frames */

    for(i=0; i<s->picture_count; i++){

        if(s->picture[i].data[0] && !s->picture[i].reference

           && s->picture[i].owner2 == s

           && (remove_current || &s->picture[i] != s->current_picture_ptr)

           /*&& s->picture[i].type!=FF_BUFFER_TYPE_SHARED*/){

            free_frame_buffer(s, &s->picture[i]);

        }

    }

}
