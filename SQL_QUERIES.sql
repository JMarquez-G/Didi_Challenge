/* Preprocessing SQL*/
/*Drop 2091 year registers from,local_create_time column 
Replace 2091 TO 2021 was used previously but the set of new date were future dates and not past
*/

UPDATE service_ticket_global
SET local_create_time =REPLACE(local_create_time,'2091','2021');

/*compensate_service_ticket_table Left JOIN */
/*Join both tables into a single one*/
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

/*1.0 Top 5 of cities that have highest average number of tickets the last three weeks*/
/*According with local_create_time_compensate column*/
SELECT city_id, name_en, (count(ticket_id)/21) AS Avg_tickets_day, (count(ticket_id)/3) AS Avg_tickets_week, count(ticket_id) AS Total_Tickets
FROM compensate_service_ticket_table
WHERE local_create_time_compensate>=(SELECT date((SELECT max(local_create_time_compensate) FROM compensate_service_ticket_table),'-21 days'))
GROUP BY city_id
ORDER BY Total_Tickets DESC LIMIT 5;

/*According with local_create_time_service_ticket column*/
SELECT city_id, name_en, (count(ticket_id)/21) AS Avg_tickets_day, (count(ticket_id)/3) AS Avg_tickets_week, count(ticket_id) AS Total_Tickets
FROM compensate_service_ticket_table
WHERE local_create_time_service_ticket>=(SELECT date((SELECT max(local_create_time_service_ticket) FROM compensate_service_ticket_table),'-21 days'))
GROUP BY city_id
ORDER BY Total_Tickets DESC LIMIT 5;


/* 2.0 day of the week there are usually more tickets on average in the top 10 cities*/
/*Use SQL to discover which day of the week there are usually more tickets on average in the top 10 cities*/
SELECT city_id, name_en, Raised_ticket_day, max(Total_Tickets) AS Max_Tickets 
FROM 
(
SELECT city_id, name_en,
CASE CAST (strftime('%w', local_create_time_service_ticket) as integer)
WHEN 0 THEN 'Sunday'
WHEN 1 THEN 'Monday'
WHEN 2 THEN 'Tuesday'
WHEN 3 THEN 'Wednesday'
WHEN 4 THEN 'Thursday'
WHEN 5 THEN 'Friday'
ELSE 'Saturday'
END AS 'Raised_ticket_day',
count(ticket_id) AS Total_Tickets
FROM compensate_service_ticket_table
GROUP BY name_en,Raised_ticket_day
ORDER BY Total_Tickets DESC
)
GROUP BY name_en
ORDER BY Max_Tickets DESC LIMIT 10;



/*3.0 How was the percentage of growth of the amount of tickets week over week for the last four weeks of the data?*/


SELECT *,lag(Total_Tickets,1,0) OVER (ORDER BY Raised_ticket_week ASC) AS Previous_total_tickets,
(Total_Tickets-(lag(Total_Tickets,1,0) OVER (ORDER BY Raised_ticket_week ASC))) AS Difference,
(((Total_Tickets-(lag(Total_Tickets,1,0) OVER (ORDER BY Raised_ticket_week ASC)))/Total_Tickets))*100 AS Percentage_growth
FROM(
SELECT CAST (strftime('%W', local_create_time_service_ticket) as INTEGER) AS Raised_ticket_week,
count(ticket_id) AS Total_Tickets
FROM compensate_service_ticket_table
WHERE 
Raised_ticket_week>=(SELECT max(CAST (strftime('%W', local_create_time_service_ticket) as integer))FROM compensate_service_ticket_table)-4
GROUP BY Raised_ticket_week
ORDER BY Raised_ticket_week DESC
);


/* 4.0 the sum of the price compensations and the number of tickets */
SELECT date(local_create_time_service_ticket) AS date_service_ticket , count(ticket_id) AS Total_tickets,
sum(price) AS Total_price
FROM compensate_service_ticket_table
GROUP BY date_service_ticket
ORDER BY date_service_ticket ASC;
