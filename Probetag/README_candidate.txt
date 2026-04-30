You are a data scientist at the company "RaceCar Insurance Experts GmbH".
Your company gets thousands of invoices from car repair shops every day.
Each of these invoices contain one or multiple cost positions,
which are individual spare parts or labor processes.
To be able to quickly process cost positions, 
your company uses a system that assigns a booking code to each individual cost position.
The purpose of these booking codes is to serve as a comprehensive internal classification of individual cost positions.

A cost position contains the following information:

PartNumber: A car-repair shop specific identifier for the cost position.
The same cost position can have different part numbers if the invoices were issued by different repair shops.

Description: A text description of the cost position.

Count: Amount of the Part that were used / Time the process took.

SumPrice: Summed price of the cost position.

BookingCode: The internal booking code that was assigned to this particular cost position.

DocumentId: The ID of the invoice that the cost position belongs to.


Your task is to build a Model that automates the assignment of booking codes to cost positions.
To this end, you were provided a dataset of digitalized cost positions and their respective booking codes.


