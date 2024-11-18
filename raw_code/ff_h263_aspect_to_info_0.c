av_const int ff_h263_aspect_to_info(AVRational aspect){

    int i;



    if(aspect.num==0) aspect= (AVRational){1,1};



    for(i=1; i<6; i++){

        if(av_cmp_q(ff_h263_pixel_aspect[i], aspect) == 0){

            return i;

        }

    }



    return FF_ASPECT_EXTENDED;

}
