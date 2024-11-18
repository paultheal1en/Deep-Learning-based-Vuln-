char *string_output_get_string(StringOutputVisitor *sov)

{

    char *string = g_string_free(sov->string, false);

    sov->string = NULL;

    return string;

}
