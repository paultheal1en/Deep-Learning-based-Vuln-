static QError *qerror_from_info(const char *fmt, va_list *va)

{

    QError *qerr;



    qerr = qerror_new();

    loc_save(&qerr->loc);



    qerr->error = error_obj_from_fmt_no_fail(fmt, va);

    qerr->err_msg = qerror_format(fmt, qerr->error);



    return qerr;

}
