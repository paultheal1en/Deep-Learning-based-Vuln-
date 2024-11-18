void string_output_visitor_cleanup(StringOutputVisitor *sov)

{

    if (sov->string) {

        g_string_free(sov->string, true);

    }



    g_list_foreach(sov->ranges, free_range, NULL);

    g_list_free(sov->ranges);

    g_free(sov);

}
