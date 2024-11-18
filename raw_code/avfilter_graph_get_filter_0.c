AVFilterContext *avfilter_graph_get_filter(AVFilterGraph *graph, char *name)

{

    int i;



    if(!name)

        return NULL;



    for(i = 0; i < graph->filter_count; i ++)

        if(graph->filters[i]->name && !strcmp(name, graph->filters[i]->name))

            return graph->filters[i];



    return NULL;

}
