AVFilter **av_filter_next(AVFilter **filter)

{

    return filter ? ++filter : &registered_avfilters[0];

}
