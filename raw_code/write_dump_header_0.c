static void write_dump_header(DumpState *s, Error **errp)

{

     Error *local_err = NULL;



    if (s->dump_info.d_class == ELFCLASS32) {

        create_header32(s, &local_err);

    } else {

        create_header64(s, &local_err);

    }

    if (local_err) {

        error_propagate(errp, local_err);

    }

}
