static int token_is_operator(QObject *obj, char op)

{

    const char *val;



    if (token_get_type(obj) != JSON_OPERATOR) {

        return 0;

    }



    val = token_get_value(obj);



    return (val[0] == op) && (val[1] == 0);

}
