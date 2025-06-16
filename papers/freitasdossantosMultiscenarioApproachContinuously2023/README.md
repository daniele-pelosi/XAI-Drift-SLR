# README #

### Paper ###
A Multi-scenario Approach to Continuously Learn and Understand Norm Violations [pdf](https://www.researchgate.net/publication/367635389_A_Multi-scenario_Approach_to_Continuously_Learn_and_Understand_Norm_Violations)

### Datasets ###

* Featurized
	This directory contains Wikipedia edits that were featurized. 
	Each edit either has a violation label or regular.
* Text
	This directory contains Wikipedia violations in text.
	Each violation can contain one or more of the violation classes (multi-label problem).

### Violation Labels ###

*	1 - Bad words
*	2 - Content Removal
*	3 - Gramatical Mistakes
*	4 - Insulting
*	5 - Sexual Content
*	6 - Racism
*	7 - Homophic
*	8 - Non-neutral point of view
*	9 - Bad Text
*	10 - Non Verifable
*	11 - Derrogatory Terms
*	12 - Break Wikipedia Style
*	13 - Sexism
*	14 - Not Vandalism
*	15 - Not Found
*	16 - Adding text about themselves (maybe merge with type 8 - Non neutral point of view) 
*	17 - Text not related
*	18 - Incite Violence
*	19 - Religion
*	20 - Bad numbering
*	21 - Advertising
	
	Examples:
*	1 - Use of swear words
	https://en.wikipedia.org/w/index.php?diff=326780621&oldid=326779718,3.9.4.11.1
	
*	2 - Removing content (Delete the text about an article)
	https://en.wikipedia.org/w/index.php?diff=326952467&oldid=326595858,2.9.5
	
*	3 - The edit contain gramatical mistakes
	https://en.wikipedia.org/w/index.php?diff=327970911&oldid=327970657,3.9
	
*	4 - The edit contains insuts
	https://en.wikipedia.org/w/index.php?diff=327362330&oldid=323155885,4
	
*	5 - The edit contains sexual content
	https://en.wikipedia.org/w/index.php?diff=326952467&oldid=326595858,2.9.5
	
*	6 - The edit contains RACIST language and/or expressions
	https://en.wikipedia.org/w/index.php?diff=326812313&oldid=326812203,2.6
	
*	7 - The edit contains HOMOPHOBIC language and/or expressions
	https://en.wikipedia.org/w/index.php?diff=327457233&oldid=327455939,7.9
	
*	8 - Use of a non-neutral point of view
	https://en.wikipedia.org/w/index.php?diff=327989298&oldid=327681862,9.8
	
*	9 - Use of text that is not in accordance with the expected writing style of editing an article on Wikipedia
	https://en.wikipedia.org/w/index.php?diff=329956728&oldid=328463239
	https://en.wikipedia.org/w/index.php?diff=326952467&oldid=326595858,2.9.5
	
*	10 - Adding information that is not possible to verify (ex. fake news)
	https://en.wikipedia.org/w/index.php?diff=328010563&oldid=327540421,10
	
*	11 - Use of derrogatory terms
	https://en.wikipedia.org/w/index.php?diff=326904915&oldid=326904850,4.11
	
*	12 - Edit the article not following Wikipedia hyper text style
	https://en.wikipedia.org/w/index.php?diff=327724846&oldid=327676605,12.9
	
*	13 - The edit contains SEXIST language and/or expressions
	https://en.wikipedia.org/w/index.php?diff=327459797&oldid=327459706,13.3
	
*	14 - The edit is not vandalism, it was wrongly classified by the mechanical turkeys
	https://en.wikipedia.org/w/index.php?diff=327292924&oldid=327292819,14
	
*	15 - The edit does not exist anymore
	
*	16 - User edits an article by adding text about themselves
	https://en.wikipedia.org/w/index.php?diff=327352246&oldid=327350306,16
	
*	17 - Addition of text not related to the article
	https://en.wikipedia.org/w/index.php?diff=326935330&oldid=326934727,17
	
*	18 - The edit incites violence
	https://en.wikipedia.org/w/index.php?diff=326780621&oldid=326779718,3.9.4.11.1
	
*	19 - The edit contains discrimination based on RELIGION
	https://en.wikipedia.org/w/index.php?diff=327520301&oldid=327520116,9.19
	
*	20 - Use of numbering that is not in accordance with the expected numbering style on Wikipedia
	https://en.wikipedia.org/w/index.php?diff=326780435&oldid=326745884,20.9
	
*	21 - Text that is advertasing
	https://en.wikipedia.org/w/index.php?diff=326975474&oldid=325719628,21



### Citation ###
