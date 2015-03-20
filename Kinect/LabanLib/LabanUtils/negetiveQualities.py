NQ = {
    'Strong':['Free+Light','Passive'],
    'Advance': ['Retreat', 'Twist+Back'],
    'Retreat': ['Advance'],
    'Condense+Enclose': ['Spread'],
    'Bind': ['Free+Light'],
    'Twist+Back': ['Advance','Direct'],
    'Spread': ['Condense+Enclose','ArmsToUpperBody'],
    'Free+Light': ['Bind','Strong'],
    'Up+Rise': ['Sink','Twist+Back'],
    'Passive': ['Strong'],
    'Jump':['Sink'],
    'ArmsToUpperBody':['Spread'],
    'Sink': ['Up+Rise','Jump'],
    'HeadDrop': ['Up+Rise'],
    'Sudden':['Passive'],
    'Rotation':['Advance','Sink'],
    'Rhythmicity':['Passive'],
    'Direct':['Twist+Back']
}

disjointQualities = {
    'Strong':['Free+Light','Passive','HeadDrop'],
    'Advance': ['Retreat','Rotation', 'Twist+Back','Passive','Sink'],
    'Retreat': ['Advance', 'Jump','Rotation'],
    'Condense+Enclose': ['Spread', 'Jump','Advance'],
    'Bind': ['Free+Light','Rhythmicity','Sudden'],
    'Twist+Back': ['Advance', 'Direct','Up+Rise','Jump'],
    'Spread': ['Condense+Enclose','ArmsToUpperBody','Sink','Passive'],
    'Free+Light': ['Bind','Strong','Passive'],
    'Up+Rise': ['Sink','Passive','Twist+Back'],
    'Passive': ['Jump','Sudden','Direct','Strong','Advance', 'Up+Rise','Rhythmicity','Rotation'],
    'Jump':['Passive', 'HeadDrop','Twist+Back','Sink','Retreat'],
    'ArmsToUpperBody':['Spread','Rhythmicity'],
    'Sink': ['Up+Rise','Jump','Advance','Rotation','Rhythmicity'],
    'HeadDrop': ['Up+Rise','Jump','Direct'],
    'Sudden':['Passive','Bind','Rhythmicity'],
    'Rotation':['Advance','Direct','Sink','Passive','Retreat'],
    'Rhythmicity':['Bind','Passive','ArmsToUpperBody','Sink','Condense+Enclose','Sudden'],
    'Direct':['Rotation', 'Twist+Back','Passive']
}
anger=['Strong','Sudden',  'Advance',  'Direct' ]
fear=['Retreat',  'Condense+Enclose',  'Bind',  'Twist+Back']
happy=['Jump',  'Rhythmicity',  'Spread', 'Free+Light',  'Up+Rise', 'Rotation']
sad=['ArmsToUpperBody', 'HeadDrop', 'Passive', 'Sink']
anger+fear+happy+sad
emotionSeparetedQualities = {
    'Strong':fear+happy+sad,
    'Advance': fear+happy+sad,
    'Retreat': anger+happy+sad,
    'Condense+Enclose': anger+happy+sad,
    'Bind': anger+happy+sad,
    'Twist+Back': anger+happy+sad,
    'Spread': anger+fear+sad,
    'Free+Light': anger+fear+sad,
    'Up+Rise':anger+fear+sad,
    'Passive': anger+fear+happy,
    'Jump':anger+fear+sad,
    'ArmsToUpperBody':anger+fear+happy,
    'Sink': anger+fear+happy,
    'HeadDrop': anger+fear+happy,
    'Sudden':fear+happy+sad,
    'Rotation':anger+fear+sad,
    'Rhythmicity':anger+fear+sad,
    'Direct':fear+happy+sad
}