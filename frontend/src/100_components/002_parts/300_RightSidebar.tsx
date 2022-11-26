import React, { useEffect, useMemo } from "react";
import { useStateControlCheckbox } from "../003_hooks/useStateControlCheckbox";
import { AnimationTypes, HeaderButton, HeaderButtonProps } from "./101_HeaderButton";
import { InputSourceSelector } from "./310_InputSourceSelector";

import { Links } from "./330_Links";

export const RightSidebar = () => {
    const sidebarAccordionEditorControllerCheckBox = useStateControlCheckbox("editor-controller");
    const sidebarAccordionLinksCheckBox = useStateControlCheckbox("links");

    const accodionButtonForEditorController = useMemo(() => {
        const accodionButtonForEditorControllerProps: HeaderButtonProps = {
            stateControlCheckbox: sidebarAccordionEditorControllerCheckBox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonForEditorControllerProps}></HeaderButton>;
    }, []);


    const accodionButtonForLinks = useMemo(() => {
        const accodionButtonForLinksProps: HeaderButtonProps = {
            stateControlCheckbox: sidebarAccordionLinksCheckBox,
            tooltip: "Open/Close",
            onIcon: ["fas", "caret-up"],
            offIcon: ["fas", "caret-up"],
            animation: AnimationTypes.spinner,
            tooltipClass: "tooltip-right",
        };
        return <HeaderButton {...accodionButtonForLinksProps}></HeaderButton>;
    }, []);


    useEffect(() => {
        sidebarAccordionEditorControllerCheckBox.updateState(true);
    }, []);

    return (
        <>
            <div className="right-sidebar">
                {sidebarAccordionEditorControllerCheckBox.trigger}
                <div className="sidebar-partition">
                    <div className="sidebar-header">
                        <div className="sidebar-header-title">Setting</div>
                        <div className="sidebar-header-caret"> {accodionButtonForEditorController}</div>
                    </div>
                    <InputSourceSelector />
                </div>

                {/* <div className="sidebar-content-row-button" onClick={() => {
                    frontendManagerState.stateControls.generalDialogCheckbox.updateState(true)
                }}>
                    edit
                </div> */}

                {sidebarAccordionLinksCheckBox.trigger}
                <div className="sidebar-partition">
                    <div className="sidebar-header">
                        <div className="sidebar-header-title">Links</div>
                        <div className="sidebar-header-caret"> {accodionButtonForLinks}</div>
                    </div>
                    <Links></Links>
                </div>
            </div>
        </>
    );
};
