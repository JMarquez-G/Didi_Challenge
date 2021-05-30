/* Preprocessing SQL*/
/*Drop 2091 year registers from,local_create_time column 
Replace 2091 TO 2021 was used previously but the set of new date were future dates and not past
*/

UPDATE service_ticket_global
SET local_create_time =REPLACE(local_create_time,'2091','2021');

/*compensate_service_ticket_table Left JOIN */
SELECT service_ticket_global.ticket_id, service_ticket_global.country_code, service_ticket_global.city_id, service_ticket_global.name_en, service_ticket_global.business_type,service_ticket_global.channel_name, service_ticket_global.subject ,
compensate_info_global_increment.price,
strftime(service_ticket_global.local_create_time) AS local_create_time_service_ticket,
strftime(compensate_info_global_increment.local_create_time) AS local_create_time_compensate, 
service_ticket_global.status
FROM service_ticket_global
LEFT JOIN
compensate_info_global_increment
ON service_ticket_global.ticket_id=compensate_info_global_increment.tticket
WHERE service_ticket_global.local_create_time NOT REGEXP '^2091' ; /*DROP 2091 RESGISTERS*/
