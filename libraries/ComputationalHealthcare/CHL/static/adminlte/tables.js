/**
 * Created by aub3 on 3/17/15.
 */


function InitializeTables(){
    $('.dataTables-proto_file_table').dataTable({
        responsive: true,
        "order": [[ 3, "desc" ]],
    });
    $('.dataTables-split_file_table').dataTable({
        responsive: true,
        "order": [[ 4, "desc" ]],
    });
    $('.dataTables-home').dataTable({
        responsive: true,
        "order": [[ 3, "desc" ]],
        "bFilter": false,
        "bLengthChange": false
    });
    $('.dataTables-dict').dataTable({
        responsive: true,
        "bFilter": false,
        "bPaginate": false,
        "bInfo":false
    });
    $('.dataTables-full').dataTable({
        responsive: true,
    });
    $('.dataTables-subset').dataTable({
        responsive: true,
        "order": [[ 0, "asc" ]],
    });
    $('.dataTables-non-export').dataTable({
        "bFilter": false,
    });
}