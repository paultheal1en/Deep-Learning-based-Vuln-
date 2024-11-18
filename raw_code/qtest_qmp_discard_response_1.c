void qtest_qmp_discard_response(QTestState *s, const char *fmt, ...)

{

    va_list ap;



    va_start(ap, fmt);

    qtest_qmpv_discard_response(s, fmt, ap);

    va_end(ap);

}
