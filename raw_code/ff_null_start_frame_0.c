void ff_null_start_frame(AVFilterLink *link, AVFilterBufferRef *picref)

{

    ff_start_frame(link->dst->outputs[0], picref);

}
